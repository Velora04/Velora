"""
Sistema de memoria de trabajo para VELORA.

Implementa mecanismos de almacenamiento y recuperación
de información para mantener contexto entre operaciones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Any


class WorkingMemory(nn.Module):
    """
    Sistema de memoria de trabajo para VELORA.
    
    Implementa un mecanismo de memoria key-value con operaciones
    de lectura y escritura basadas en atención, permitiendo mantener
    contexto entre diferentes pasos de procesamiento.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_size: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Inicializar memoria
        memory_tensor = torch.zeros(1, memory_size, hidden_dim)
        self.register_buffer("memory", memory_tensor)
        
        # Proyecciones para acceso a memoria
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Proyecciones para operaciones de memoria
        self.write_gate = nn.Linear(hidden_dim * 2, 1)
        self.erase_gate = nn.Linear(hidden_dim, memory_size)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Parámetros de memoria
        self.memory_decay = nn.Parameter(torch.ones(1, 1, memory_size) * 0.95)
        
        # Normalización y dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos de las capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def reset_memory(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """
        Reinicia el estado de la memoria.
        
        Args:
            batch_size: Tamaño del batch para la nueva memoria
            device: Dispositivo para la nueva memoria
        """
        if device is None and self.memory is not None:
            device = self.memory.device
            
        self.memory = torch.zeros(
            batch_size, self.memory_size, self.hidden_dim, 
            device=device
        )
    
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lee información de la memoria utilizando atención.
        
        Args:
            query: Tensor de consulta [batch_size, seq_len, hidden_dim]
            
        Returns:
            Información recuperada y pesos de atención
        """
        batch_size, seq_len, _ = query.size()
        
        # Proyectar query para lectura
        q = self.query_proj(query)
        
        # Si la memoria no está inicializada o el batch size no coincide, reiniciarla
        if self.memory is None or self.memory.size(0) != batch_size:
            self.reset_memory(batch_size, query.device)
        
        # Proyectar memoria para operación de atención
        k = self.key_proj(self.memory)
        v = self.value_proj(self.memory)
        
        # Dividir en múltiples cabezas de atención
        head_dim = self.hidden_dim // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, self.memory_size, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, self.memory_size, self.num_heads, head_dim).transpose(1, 2)
        
        # Calcular scores de atención: (B, H, Lq, D) x (B, H, D, Lk) -> (B, H, Lq, Lk)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        
        # Aplicar softmax para obtener pesos de atención
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aplicar pesos para obtener contexto: (B, H, Lq, Lk) x (B, H, Lk, D) -> (B, H, Lq, D)
        context = torch.matmul(attention_weights, v)
        
        # Transponer y reshapear: (B, H, Lq, D) -> (B, Lq, H*D)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Concatenar con query original para enriquecer contexto
        combined = torch.cat([query, context], dim=-1)
        
        # Proyectar a dimensión original
        read_output = self.output_proj(combined)
        read_output = self.layer_norm(read_output)
        
        # Devolver salida y pesos de atención (útiles para análisis y visualización)
        # Promediamos los pesos entre cabezas para simplificar
        mean_attention = attention_weights.mean(dim=1)
        
        return read_output, mean_attention
    
    def write(self, input_data: torch.Tensor, memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Escribe información en la memoria y actualiza su estado.
        
        Args:
            input_data: Tensor con información a escribir [batch_size, seq_len, hidden_dim]
            memory_mask: Máscara opcional para seleccionar slots de memoria [batch_size, memory_size]
            
        Returns:
            Memoria actualizada y metadatos de la operación
        """
        batch_size, seq_len, _ = input_data.size()
        
        # Si la memoria no está inicializada o el batch size no coincide, reiniciarla
        if self.memory is None or self.memory.size(0) != batch_size:
            self.reset_memory(batch_size, input_data.device)
        
        # Primero leer memoria para contexto
        context, attention_read = self.read(input_data)
        
        # Determinar qué escribir mediante una puerta
        combined = torch.cat([input_data, context], dim=-1)
        write_gate_vals = torch.sigmoid(self.write_gate(combined))  # [batch_size, seq_len, 1]
        
        # Determinar dónde escribir 
        # Promediamos la entrada a lo largo de la dimensión de secuencia para tener una representación por batch
        input_avg = input_data.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Computar distribución de escritura sobre slots de memoria
        write_weights = F.softmax(self.erase_gate(input_avg), dim=-1)  # [batch_size, memory_size]
        
        # Si se proporciona una máscara, aplicarla
        if memory_mask is not None:
            write_weights = write_weights * memory_mask
            # Renormalizar
            write_weights = write_weights / (write_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Expandir dimensiones para facilitar operaciones
        write_weights = write_weights.unsqueeze(1)  # [batch_size, 1, memory_size]
        
        # Computar valor a escribir (promedio ponderado de la secuencia)
        write_gate_expanded = write_gate_vals.transpose(1, 2)  # [batch_size, 1, seq_len]
        write_data = torch.bmm(write_gate_expanded, input_data)  # [batch_size, 1, hidden_dim]
        
        # Aplicar decaimiento a memoria existente
        self.memory = self.memory * self.memory_decay
        
        # Escribir en memoria:
        # 1. Para cada slot afectado, interpolamos entre memoria actual y nuevo valor
        # 2. La ponderación viene dada por write_weights
        memory_update = torch.bmm(write_weights.transpose(1, 2), write_data)  # [batch_size, memory_size, hidden_dim]
        
        # Actualizar memoria
        self.memory = self.memory + memory_update
        
        # Normalizar memoria para estabilidad numérica
        self.memory = F.normalize(self.memory, p=2, dim=-1) * math.sqrt(self.hidden_dim)
        
        # Crear metadatos para análisis
        metadata = {
            "write_gate": write_gate_vals.mean().item(),
            "write_weights": write_weights.squeeze(1),
            "memory_usage": (write_weights.sum(dim=-1) / self.memory_size).mean().item()
        }
        
        return self.memory, metadata
    
    def forward(self, 
                input_data: torch.Tensor, 
                mode: str = "read_write",
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Operación completa de memoria: leer y/o escribir.
        
        Args:
            input_data: Tensor de entrada [batch_size, seq_len, hidden_dim]
            mode: Modo de operación ("read", "write", o "read_write")
            memory_mask: Máscara opcional para memoria
            
        Returns:
            Resultado de la operación y metadatos
        """
        metadata = {}
        
        if mode == "read" or mode == "read_write":
            read_output, attention_weights = self.read(input_data)
            metadata["read_attention"] = attention_weights
            
            if mode == "read":
                return read_output, metadata
        
        if mode == "write" or mode == "read_write":
            memory, write_metadata = self.write(input_data, memory_mask)
            metadata.update(write_metadata)
            
            if mode == "write":
                # En modo sólo escritura, devolvemos la entrada original
                return input_data, metadata
        
        # En modo read_write, devolvemos el resultado de lectura
        return read_output, metadata


class ContextManager(nn.Module):
    """
    Gestor de contexto para VELORA.
    
    Mantiene continuidad contextual entre múltiples operaciones,
    gestiona referencias y dependencias, y proporciona mecanismos
    para diferentes niveles de persistencia.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_size: int = 128,
        num_contexts: int = 3,  # Número de niveles de contexto (corto, medio, largo plazo)
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.num_contexts = num_contexts
        
        # Memorias para diferentes escalas temporales
        self.memories = nn.ModuleList([
            WorkingMemory(
                hidden_dim=hidden_dim,
                memory_size=memory_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_contexts)
        ])
        
        # Factores de decaimiento para diferentes contextos
        decay_rates = torch.tensor([0.9, 0.97, 0.995])  # Corto, medio y largo plazo
        self.register_buffer("decay_rates", decay_rates)
        
        # Puerta para seleccionar qué contexto utilizar
        self.context_gate = nn.Linear(hidden_dim, num_contexts)
        
        # Variables para seguimiento de estado
        self.session_active = False
        self.interaction_count = 0
        
    def start_session(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """
        Inicia una nueva sesión de interacción.
        
        Args:
            batch_size: Tamaño del batch para esta sesión
            device: Dispositivo para tensores
        """
        # Resetear todas las memorias
        for memory in self.memories:
            memory.reset_memory(batch_size, device)
        
        # Marcar sesión como activa y resetear contador
        self.session_active = True
        self.interaction_count = 0
    
    def end_session(self):
        """Finaliza la sesión actual y libera recursos."""
        self.session_active = False
        # Permitir que el recolector de basura libere memorias
        for memory in self.memories:
            memory.memory = None
    
    def forward(self, 
                input_data: torch.Tensor,
                context_type: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Procesa entrada a través del gestor de contexto.
        
        Args:
            input_data: Tensor de entrada [batch_size, seq_len, hidden_dim]
            context_type: Tipo específico de contexto a utilizar (opcional)
            
        Returns:
            Representación contextualizada y metadatos
        """
        batch_size, seq_len, _ = input_data.size()
        
        # Iniciar sesión si no está activa
        if not self.session_active:
            self.start_session(batch_size, input_data.device)
        
        # Incrementar contador de interacciones
        self.interaction_count += 1
        
        # Si no se especifica tipo de contexto, calcular pesos para cada uno
        if context_type is None:
            # Obtener representación promedio de entrada
            input_avg = input_data.mean(dim=1)
            
            # Calcular pesos para cada contexto
            context_weights = F.softmax(self.context_gate(input_avg), dim=1)
        else:
            # Usar one-hot para el contexto especificado
            context_weights = torch.zeros(batch_size, self.num_contexts, device=input_data.device)
            context_weights[:, context_type] = 1.0
        
        # Procesar a través de cada memoria contextual
        context_outputs = []
        all_metadata = {}
        
        for i, memory in enumerate(self.memories):
            # Leer de esta memoria
            context_output, metadata = memory.forward(input_data, mode="read_write")
            context_outputs.append(context_output)
            all_metadata[f"context_{i}"] = metadata
        
        # Combinar salidas según pesos de contexto
        # Expandir dimensiones para multiplicación por batch
        context_weights = context_weights.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, num_contexts, 1]
        
        # Apilar outputs de contexto: [batch_size, seq_len, num_contexts, hidden_dim]
        stacked_outputs = torch.stack(context_outputs, dim=2)
        
        # Aplicar pesos y sumar: [batch_size, seq_len, hidden_dim]
        weighted_output = (stacked_outputs * context_weights).sum(dim=2)
        
        # Añadir metadata
        all_metadata["context_weights"] = context_weights.squeeze(1).squeeze(2)
        all_metadata["interaction_count"] = self.interaction_count
        
        return weighted_output, all_metadata