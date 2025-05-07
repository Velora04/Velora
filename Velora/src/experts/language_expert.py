"""
Experto en lenguaje para VELORA.

Este módulo implementa el experto especializado en procesamiento de lenguaje natural.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageExpert(nn.Module):
    """
    Experto en procesamiento de lenguaje natural.
    
    Capaz de:
    1. Clasificar tipos de consultas
    2. Analizar intención del usuario
    3. Generar respuestas coherentes
    4. Procesar secuencias de texto
    """
    
    def __init__(self, config):
        super(LanguageExpert, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        
        # Procesador de secuencias
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Capas de atención para procesar secuencias
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Capas feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Capas de normalización
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # Clasificador de tipo de consulta
        self.query_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)  # pregunta, comando, declaración
        )
        
        # Predictor de tokens (para completar frases)
        self.token_predictor = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Proyector de salida
        self.output_projector = nn.Linear(self.hidden_dim, self.input_dim)
        
    def forward(self, x, task_hint=None):
        """
        Procesa entrada lingüística.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_dim]
            task_hint: Pista sobre la tarea a realizar
            
        Returns:
            Representación procesada y metadatos
        """
        # Verificar formato de entrada
        if x.dim() < 3:
            # Si no es secuencia, expandir a formato secuencial
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Codificar secuencia
        encoded = self.encoder(x)
        
        # Aplicar atención
        attn_output, attn_weights = self.self_attention(encoded, encoded, encoded)
        residual1 = self.norm1(encoded + attn_output)
        
        # Aplicar feed-forward
        ff_output = self.feed_forward(residual1)
        hidden = self.norm2(residual1 + ff_output)
        
        # Clasificar tipo de consulta (usando primera token como [CLS])
        query_logits = self.query_classifier(hidden[:, 0, :])
        query_type = torch.argmax(query_logits, dim=1)
        
        # Si se proporciona hint de tarea, usarlo
        if task_hint is not None:
            query_type = task_hint if isinstance(task_hint, torch.Tensor) else torch.tensor([task_hint], device=x.device)
        
        # Predecir tokens siguientes
        token_logits = self.token_predictor(hidden)
        next_tokens = torch.argmax(token_logits, dim=2)
        
        # Calcular representación de salida (pooling de la secuencia)
        output_repr = torch.mean(hidden, dim=1)
        
        # Proyectar al espacio de salida
        output = self.output_projector(output_repr)
        
        return output, {
            'query_type': query_type,
            'query_logits': query_logits,
            'token_logits': token_logits,
            'next_tokens': next_tokens,
            'attention_weights': attn_weights,
            'hidden_states': hidden
        }
    
    def predict_next_tokens(self, hidden_states, num_tokens=5):
        """
        Predice los próximos tokens basado en el estado oculto.
        
        Args:
            hidden_states: Estados ocultos del modelo
            num_tokens: Número de tokens a predecir
            
        Returns:
            Secuencia de tokens predichos
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Inicializar con el último estado
        current_hidden = hidden_states[:, -1:, :]
        
        # Lista para tokens predichos
        predicted_tokens = []
        
        # Generar tokens autoregressivamente
        for _ in range(num_tokens):
            # Procesar con capas de atención y feed-forward
            attn_output, _ = self.self_attention(
                current_hidden, current_hidden, current_hidden
            )
            residual1 = self.norm1(current_hidden + attn_output)
            
            ff_output = self.feed_forward(residual1)
            next_hidden = self.norm2(residual1 + ff_output)
            
            # Predecir siguiente token
            logits = self.token_predictor(next_hidden[:, -1, :])
            next_token = torch.argmax(logits, dim=1, keepdim=True)
            
            # Guardar token
            predicted_tokens.append(next_token)
            
            # Crear embedding para el nuevo token
            # (simplificado - en una implementación real usaríamos un embedding)
            token_embedding = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
            
            # Actualizar contexto para siguiente predicción
            current_hidden = torch.cat([current_hidden, token_embedding], dim=1)
        
        # Combinar tokens en una secuencia
        return torch.cat(predicted_tokens, dim=1)
    
    def explain(self, metadata):
        """
        Genera explicación del procesamiento lingüístico.
        
        Args:
            metadata: Metadatos del procesamiento
            
        Returns:
            Explicación en formato de texto
        """
        query_types = ['pregunta', 'comando', 'declaración']
        
        explanations = []
        
        for i in range(len(metadata['query_type'])):
            query_idx = metadata['query_type'][i].item()
            
            # Extraer información para explicación
            explanation = f"Tipo de consulta: {query_types[query_idx]}\n"
            
            # Añadir detalles sobre tokens predichos
            if 'next_tokens' in metadata:
                next_tokens = metadata['next_tokens'][i]
                explanation += f"Tokens predichos: {next_tokens[:5].tolist()}\n"
            
            # Añadir información sobre atención
            if 'attention_weights' in metadata:
                attn_sum = torch.sum(metadata['attention_weights'][i]).item()
                explanation += f"Intensidad de atención: {attn_sum:.2f}"
            
            explanations.append(explanation)
        
        return explanations