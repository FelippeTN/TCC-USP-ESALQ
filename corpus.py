"""Corpus sintético de ferramentas (tool schemas), com pares confusáveis propositais.

Um par confusável = duas ferramentas com propósito quase idêntico (ex.: send_message vs
send_message_v2, get_user vs get_user_profile) para que a métrica de acurácia meça
discriminação fina, não só "achou a categoria certa".
"""
import random

from config import CORE_DOMAINS

# (nome, descrição, params) por domínio. Cada domínio tem pares confusáveis intercalados.
_TOOL_TEMPLATES = {
    "messages": [
        ("send_message", "Envia uma mensagem de texto para um usuário", {"to": "string", "text": "string"}),
        ("send_message_v2", "Envia uma mensagem de texto para um usuário, com suporte a anexos", {"to": "string", "text": "string", "attachment_id": "string"}),
        ("list_messages", "Lista as mensagens de uma conversa", {"conversation_id": "string", "limit": "integer"}),
        ("search_messages", "Busca mensagens por palavra-chave", {"query": "string", "conversation_id": "string"}),
        ("delete_message", "Remove uma mensagem enviada", {"message_id": "string"}),
        ("edit_message", "Edita o conteúdo de uma mensagem já enviada", {"message_id": "string", "text": "string"}),
        ("mark_message_read", "Marca uma mensagem como lida", {"message_id": "string"}),
        ("mark_conversation_read", "Marca todas as mensagens de uma conversa como lidas", {"conversation_id": "string"}),
        ("forward_message", "Reenvia uma mensagem existente para outro usuário", {"message_id": "string", "to": "string"}),
        ("pin_message", "Fixa uma mensagem no topo da conversa", {"message_id": "string"}),
        ("react_to_message", "Adiciona uma reação (emoji) a uma mensagem", {"message_id": "string", "emoji": "string"}),
        ("schedule_message", "Agenda o envio de uma mensagem para um horário futuro", {"to": "string", "text": "string", "send_at": "string"}),
        ("export_conversation", "Exporta o histórico de uma conversa em arquivo", {"conversation_id": "string", "format": "string"}),
    ],
    "users": [
        ("get_user", "Retorna os dados básicos de um usuário", {"user_id": "string"}),
        ("get_user_profile", "Retorna o perfil completo de um usuário, incluindo preferências", {"user_id": "string"}),
        ("list_users", "Lista usuários de um grupo ou organização", {"group_id": "string", "limit": "integer"}),
        ("search_users", "Busca usuários por nome ou e-mail", {"query": "string"}),
        ("update_user", "Atualiza dados cadastrais de um usuário", {"user_id": "string", "fields": "object"}),
        ("deactivate_user", "Desativa a conta de um usuário", {"user_id": "string"}),
        ("block_user", "Bloqueia um usuário para o usuário atual", {"user_id": "string"}),
        ("unblock_user", "Remove o bloqueio de um usuário", {"user_id": "string"}),
        ("get_user_status", "Retorna o status online/offline de um usuário", {"user_id": "string"}),
        ("invite_user", "Convida um novo usuário para a organização", {"email": "string"}),
        ("remove_user", "Remove permanentemente um usuário da organização", {"user_id": "string"}),
        ("list_user_groups", "Lista os grupos aos quais um usuário pertence", {"user_id": "string"}),
        ("reset_user_password", "Envia um link de redefinição de senha para o usuário", {"user_id": "string"}),
    ],
    "groups": [
        ("create_group", "Cria um novo grupo de conversa", {"name": "string", "member_ids": "array"}),
        ("delete_group", "Remove um grupo existente", {"group_id": "string"}),
        ("list_groups", "Lista os grupos de uma organização", {"org_id": "string"}),
        ("get_group", "Retorna os dados básicos de um grupo", {"group_id": "string"}),
        ("get_group_details", "Retorna os dados completos de um grupo, incluindo configurações", {"group_id": "string"}),
        ("add_group_member", "Adiciona um usuário a um grupo", {"group_id": "string", "user_id": "string"}),
        ("remove_group_member", "Remove um usuário de um grupo", {"group_id": "string", "user_id": "string"}),
        ("rename_group", "Renomeia um grupo", {"group_id": "string", "name": "string"}),
        ("mute_group", "Silencia notificações de um grupo", {"group_id": "string"}),
        ("archive_group", "Arquiva um grupo sem excluí-lo", {"group_id": "string"}),
        ("list_group_members", "Lista os membros de um grupo", {"group_id": "string"}),
        ("transfer_group_ownership", "Transfere a titularidade de um grupo para outro membro", {"group_id": "string", "new_owner_id": "string"}),
        ("set_group_permissions", "Define permissões de membros de um grupo", {"group_id": "string", "permissions": "object"}),
    ],
    "notifications": [
        ("send_notification", "Envia uma notificação push para um usuário", {"user_id": "string", "text": "string"}),
        ("send_notification_batch", "Envia a mesma notificação para uma lista de usuários", {"user_ids": "array", "text": "string"}),
        ("list_notifications", "Lista as notificações recentes de um usuário", {"user_id": "string", "limit": "integer"}),
        ("mark_notification_read", "Marca uma notificação como lida", {"notification_id": "string"}),
        ("delete_notification", "Remove uma notificação", {"notification_id": "string"}),
        ("get_notification_settings", "Retorna as preferências de notificação de um usuário", {"user_id": "string"}),
        ("update_notification_settings", "Atualiza as preferências de notificação de um usuário", {"user_id": "string", "settings": "object"}),
        ("mute_notifications", "Silencia todas as notificações de um usuário por um período", {"user_id": "string", "duration_minutes": "integer"}),
        ("unmute_notifications", "Reativa as notificações de um usuário", {"user_id": "string"}),
        ("schedule_notification", "Agenda uma notificação para envio futuro", {"user_id": "string", "text": "string", "send_at": "string"}),
        ("cancel_scheduled_notification", "Cancela uma notificação agendada", {"notification_id": "string"}),
        ("get_notification_status", "Retorna o status de entrega de uma notificação", {"notification_id": "string"}),
        ("resend_notification", "Reenvia uma notificação que falhou", {"notification_id": "string"}),
    ],
}

# pares (nome_a, nome_b) que são propositalmente confusáveis, usados p/ análise de discriminação
CONFUSABLE_PAIRS = [
    ("send_message", "send_message_v2"),
    ("get_user", "get_user_profile"),
    ("mark_message_read", "mark_conversation_read"),
    ("block_user", "unblock_user"),
    ("get_group", "get_group_details"),
    ("mute_group", "archive_group"),
    ("send_notification", "send_notification_batch"),
    ("mute_notifications", "unmute_notifications"),
]


_PAIRED = {n for pair in CONFUSABLE_PAIRS for n in pair}


def _all_tools_for_domain(domain: str) -> list[dict]:
    """Ferramentas do domínio, com os pares confusáveis primeiro.

    A ordem importa: build_toolset corta por prefixo, então pares confusáveis
    precisam sobreviver até no menor toolset (size=10), senão a acurácia sobe
    artificialmente e a métrica deixa de medir discriminação fina.
    """
    tools = [
        {"name": name, "description": desc, "domain": domain, "parameters": params}
        for name, desc, params in _TOOL_TEMPLATES[domain]
    ]
    tools.sort(key=lambda t: t["name"] not in _PAIRED)  # estável: pares primeiro, resto na ordem original
    return tools


def build_toolset(size: int, domains: list[str] | None = None) -> list[dict]:
    """Monta um toolset sintético de `size` ferramentas, distribuído entre os domínios.

    Distribui o mais uniformemente possível; sobras vão para os primeiros domínios.
    """
    domains = domains or CORE_DOMAINS
    per_domain, remainder = divmod(size, len(domains))

    pool = []
    for i, domain in enumerate(domains):
        available = _all_tools_for_domain(domain)
        n = per_domain + (1 if i < remainder else 0)
        if n > len(available):
            raise ValueError(f"domínio '{domain}' só tem {len(available)} ferramentas, pedido {n}")
        pool.extend(available[:n])
    return pool


TOOLS_BY_NAME = {
    t["name"]: t for d in _TOOL_TEMPLATES for t in _all_tools_for_domain(d)
}


def build_toolset_for_query(size: int, query: dict, domains: list[str] | None = None) -> list[dict]:
    """Toolset de `size` ferramentas garantindo que o gabarito da query esteja dentro.

    Sem isso o eixo de exposição fica confundido: num toolset de 10 sorteado do pool
    de 52, a ferramenta correta quase nunca está presente e a acurácia cai por
    ausência, não por efeito do tamanho da exposição. Aqui o gabarito (e o `lure`,
    quando existe) são sempre incluídos e o resto é preenchido com distratores.

    O preenchimento é determinístico por query: os toolsets são aninhados
    (size 10 ⊂ 30 ⊂ 50) e idênticos entre modelos e repetições, então a comparação
    entre condições é pareada de verdade.
    """
    pool_names = [t["name"] for d in (domains or CORE_DOMAINS) for t in _all_tools_for_domain(d)]

    required = [n for n in query["expected"] if n in pool_names]
    if query.get("lure") and query["lure"] in pool_names:
        required.append(query["lure"])
    # pares confusáveis do gabarito entram junto: é o que faz a query "ambígua" ser ambígua
    for a, b in CONFUSABLE_PAIRS:
        if a in required and b not in required:
            required.append(b)
        elif b in required and a not in required:
            required.append(a)

    if len(required) > size:
        raise ValueError(f"query {query['id']} exige {len(required)} ferramentas, size={size}")

    fillers = [n for n in pool_names if n not in required]
    random.Random(f"{query['id']}|{len(pool_names)}").shuffle(fillers)  # ordem fixa → aninhamento
    chosen = required + fillers[:size - len(required)]
    return [TOOLS_BY_NAME[n] for n in chosen]


if __name__ == "__main__":
    from config import TOOLSET_SIZES

    for size in TOOLSET_SIZES:
        ts = build_toolset(size)
        assert len(ts) == size, f"size={size} gerou {len(ts)}"
        assert len({t["name"] for t in ts}) == size, "nomes duplicados no toolset"

    # pares confusáveis sobrevivem até no menor toolset
    smallest = {t["name"] for t in build_toolset(min(TOOLSET_SIZES))}
    kept = [p for p in CONFUSABLE_PAIRS if p[0] in smallest and p[1] in smallest]
    assert len(kept) >= len(CORE_DOMAINS), f"só {len(kept)} pares confusáveis em size={min(TOOLSET_SIZES)}"

    # toolsets menores são subconjuntos dos maiores (comparação entre tamanhos fica válida)
    for a, b in zip(TOOLSET_SIZES, TOOLSET_SIZES[1:]):
        assert {t["name"] for t in build_toolset(a)} <= {t["name"] for t in build_toolset(b)}

    # --- toolset por query: o gabarito tem que estar presente em TODOS os tamanhos ---
    from queries import QUERIES

    for q in QUERIES:
        prev = None
        for size in TOOLSET_SIZES:
            ts = build_toolset_for_query(size, q)
            names = {t["name"] for t in ts}
            assert len(ts) == size == len(names), f"{q['id']}/{size}: toolset com duplicatas"
            assert set(q["expected"]) <= names, f"{q['id']}/{size}: gabarito ausente do toolset"
            if q["lure"]:
                assert q["lure"] in names, f"{q['id']}/{size}: lure ausente — distrator não é testado"
            assert prev is None or prev <= names, f"{q['id']}/{size}: toolsets não são aninhados"
            prev = names

        # determinístico: mesma query, mesmo toolset em qualquer execução
        assert build_toolset_for_query(30, q) == build_toolset_for_query(30, q)

    print(f"corpus.py: ok — pool {len(TOOLS_BY_NAME)} ferramentas, {len(kept)} pares confusáveis em size={min(TOOLSET_SIZES)}, "
          f"gabarito presente em {len(QUERIES)} queries × {len(TOOLSET_SIZES)} tamanhos")
