# src/models/router.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralRouter(nn.Module):
    """
    Enrutador neuronal que determina qué experto debe procesar una entrada.
    
    Utiliza redes neuronales profundas para:
    1. Clasificar el dominio de la entrada (matemáticas/lenguaje)
    2. Determinar la tarea específica dentro del dominio
    3. Asignar pesos a cada experto para fusión de resultados
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, num_experts=2):
        super(NeuralRouter, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Codificador común para diferentes formatos de entrada
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Red de atención para secuencias
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Clasificador de dominio
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Clasificadores de tareas específicas
        self.math_task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # Tareas matemáticas: suma, resta, mult, div
        )
        
        self.language_task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # Tareas lenguaje: pregunta, comando, declaración
        )
        
        # Estimador de confianza
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Determina el enrutamiento óptimo para la entrada.
        
        Args:
            x: Entrada en formato tensor
               - Secuencial: [batch_size, seq_len, input_dim]
               - No secuencial: [batch_size, input_dim]
        
        Returns:
            Diccionario con decisiones de enrutamiento
        """
        batch_size = x.size(0)
        
        # Detectar tipo de formato (secuencial o no)
        is_sequence = (x.dim() == 3)
        
        # Codificar según formato
        if is_sequence:
            # Para secuencias, procesar con atención
            seq_len = x.size(1)
            
            # Aplicar encoder a cada token
            flattened = x.reshape(-1, self.input_dim)
            encoded_flat = self.encoder(flattened)
            encoded = encoded_flat.reshape(batch_size, seq_len, self.hidden_dim)
            
            # Aplicar mecanismo de atención
            attn_output, _ = self.attention(encoded, encoded, encoded)
            
            # Representación agregada mediante pooling
            hidden = torch.mean(attn_output, dim=1)
        else:
            # Para entradas no secuenciales, codificar directamente
            hidden = self.encoder(x)
        
        # Clasificar dominio
        domain_logits = self.domain_classifier(hidden)
        domain_probs = F.softmax(domain_logits, dim=1)
        
        # Determinar dominio primario
        primary_domain = torch.argmax(domain_probs, dim=1)
        
        # Clasificar tareas específicas
        math_task_logits = self.math_task_classifier(hidden)
        math_task_probs = F.softmax(math_task_logits, dim=1)
        math_operation = torch.argmax(math_task_probs, dim=1)
        
        language_task_logits = self.language_task_classifier(hidden)
        language_task_probs = F.softmax(language_task_logits, dim=1)
        language_task = torch.argmax(language_task_probs, dim=1)
        
        # Estimar confianza
        confidence = self.confidence_estimator(hidden)
        
        # Calcular pesos para expertos
        expert_weights = torch.zeros(batch_size, self.num_experts, device=x.device)
        
        for i in range(batch_size):
            for j in range(self.num_experts):
                # El peso de cada experto es su probabilidad de dominio
                expert_weights[i, j] = domain_probs[i, j]
        
        # Asegurar pesos normalizados
        expert_weights = F.normalize(expert_weights, p=1, dim=1)
        
        return {
            'primary_domain': primary_domain,
            'domain_logits': domain_logits,
            'domain_probs': domain_probs,
            'expert_weights': expert_weights,
            'math_operation': math_operation,
            'math_task_logits': math_task_logits,
            'math_task_probs': math_task_probs,
            'language_task': language_task,
            'language_task_logits': language_task_logits,
            'language_task_probs': language_task_probs,
            'confidence': confidence,
            'is_sequence': torch.tensor([is_sequence], device=x.device)
        }
    
    def explain_routing(self, routing_info):
        """
        Genera explicaciones sobre decisiones de enrutamiento.
        
        Args:
            routing_info: Información de enrutamiento
            
        Returns:
            Lista de explicaciones textuales
        """
        domain_names = ['matemáticas', 'lenguaje']
        math_op_names = ['suma', 'resta', 'multiplicación', 'división']
        language_task_names = ['pregunta', 'comando', 'declaración']
        
        explanations = []
        batch_size = routing_info['primary_domain'].size(0)
        
        for i in range(batch_size):
            domain_idx = routing_info['primary_domain'][i].item()
            domain_prob = routing_info['domain_probs'][i, domain_idx].item()
            conf_val = routing_info['confidence'][i].item()
            
            explanation = f"Dominio principal: {domain_names[domain_idx]} (probabilidad: {domain_prob:.2f}, confianza: {conf_val:.2f})"
            
            if domain_idx == 0:  # Matemáticas
                math_op = routing_info['math_operation'][i].item()
                explanation += f", Operación: {math_op_names[math_op]}"
            else:  # Lenguaje
                lang_task = routing_info['language_task'][i].item()
                explanation += f", Tipo de consulta: {language_task_names[lang_task]}"
            
            explanations.append(explanation)
        
        return explanations