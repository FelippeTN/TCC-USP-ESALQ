"""Dataset de queries em pt-BR — instrumento primário do experimento.

64 queries, 16 por tipo. O gabarito (`expected`) é o que define acurácia e alucinação:

- direct      → uma ferramenta claramente correta. `expected` = nome dela.
- ambiguous   → a query cabe em mais de uma ferramenta de um par confusável.
                `expected` = conjunto aceitável; acertar qualquer uma conta.
- distractor  → a ferramenta correta existe, mas o texto evoca lexicalmente outra
                (o `lure`). Mede alucinação induzida por similaridade de superfície.
- no_tool     → nenhuma ferramenta se aplica. Chamar qualquer uma = alucinação.

REVISÃO: cada item tem `reviewed_by`. Preencher com as iniciais do segundo avaliador
depois da conferência independente do gabarito (ver README, seção Validade).
"""

# (id, tipo, texto, expected, lure)
# expected: str | tuple[str, ...] | None
_QUERIES = [
    # ---------- direct (16) ----------
    ("d01", "direct", "Manda uma mensagem pro João avisando que a reunião atrasou.", "send_message", None),
    ("d02", "direct", "Lista as mensagens da conversa 8842.", "list_messages", None),
    ("d03", "direct", "Apaga a mensagem msg_5521 que eu mandei sem querer.", "delete_message", None),
    ("d04", "direct", "Corrige o texto da mensagem msg_119 para 'chego às 14h'.", "edit_message", None),
    ("d05", "direct", "Procura nas mensagens da conversa 900 onde falaram de orçamento.", "search_messages", None),
    ("d06", "direct", "Fixa a mensagem msg_77 no topo do chat.", "pin_message", None),
    ("d07", "direct", "Quais são os dados cadastrais do usuário u_301?", "get_user", None),
    ("d08", "direct", "Procura o usuário com e-mail ana.souza@empresa.com.", "search_users", None),
    ("d09", "direct", "Desativa a conta do usuário u_442, ele saiu da empresa.", "deactivate_user", None),
    ("d10", "direct", "Convida o marcos@parceiro.com para a organização.", "invite_user", None),
    ("d11", "direct", "Cria um grupo chamado 'Projeto Aurora' com a Ana e o Pedro.", "create_group", None),
    ("d12", "direct", "Adiciona o usuário u_88 no grupo g_12.", "add_group_member", None),
    ("d13", "direct", "Quem são os membros do grupo g_12?", "list_group_members", None),
    ("d14", "direct", "Renomeia o grupo g_30 para 'Financeiro 2026'.", "rename_group", None),
    ("d15", "direct", "Manda uma notificação push pro u_15 avisando do deploy.", "send_notification", None),
    ("d16", "direct", "Agenda uma notificação pro u_20 para amanhã às 9h.", "schedule_notification", None),

    # ---------- ambiguous (16) ----------
    ("a01", "ambiguous", "Envia essa mensagem pra Carla.", ("send_message", "send_message_v2"), None),
    ("a02", "ambiguous", "Manda um recado pro time com o relatório junto.", ("send_message", "send_message_v2"), None),
    ("a03", "ambiguous", "Me mostra as informações do usuário u_210.", ("get_user", "get_user_profile"), None),
    ("a04", "ambiguous", "Puxa o cadastro completo da Ana.", ("get_user", "get_user_profile"), None),
    ("a05", "ambiguous", "Marca como lida.", ("mark_message_read", "mark_conversation_read"), None),
    ("a06", "ambiguous", "Já vi tudo isso aqui, pode dar baixa.", ("mark_message_read", "mark_conversation_read"), None),
    ("a07", "ambiguous", "Não quero mais receber nada do u_99.", ("block_user", "mute_notifications"), None),
    ("a08", "ambiguous", "Libera o u_99 de novo.", ("unblock_user", "unmute_notifications"), None),
    ("a09", "ambiguous", "Me passa o grupo g_45.", ("get_group", "get_group_details"), None),
    ("a10", "ambiguous", "Quero ver como está configurado o g_45.", ("get_group", "get_group_details"), None),
    ("a11", "ambiguous", "Tira o grupo g_7 da minha lista.", ("archive_group", "delete_group", "mute_group"), None),
    ("a12", "ambiguous", "Silencia o g_7 por enquanto.", ("mute_group", "archive_group"), None),
    ("a13", "ambiguous", "Avisa o pessoal do deploy.", ("send_notification", "send_notification_batch"), None),
    ("a14", "ambiguous", "Notifica o u_15 e o u_16 sobre a manutenção.", ("send_notification_batch", "send_notification"), None),
    ("a15", "ambiguous", "Tira o u_50 daqui.", ("remove_group_member", "remove_user", "deactivate_user"), None),
    ("a16", "ambiguous", "Manda de novo, não chegou.", ("resend_notification", "forward_message", "send_message"), None),

    # ---------- distractor (16) ----------
    # o `lure` é a ferramenta errada que o texto evoca lexicalmente
    ("x01", "distractor", "Preciso reenviar pro Pedro aquela mensagem msg_44 que a Ana mandou.", "forward_message", "resend_notification"),
    ("x02", "distractor", "Essa notificação not_12 não chegou, dispara de novo.", "resend_notification", "forward_message"),
    ("x03", "distractor", "Quero exportar o histórico da conversa 8842 num arquivo.", "export_conversation", "list_messages"),
    ("x04", "distractor", "Reage com 👍 na mensagem msg_90.", "react_to_message", "pin_message"),
    ("x05", "distractor", "Programa essa mensagem pro Pedro sair só amanhã cedo.", "schedule_message", "schedule_notification"),
    ("x06", "distractor", "Programa um aviso push pro Pedro pra amanhã cedo.", "schedule_notification", "schedule_message"),
    ("x07", "distractor", "Manda o link de redefinição de senha pro u_77.", "reset_user_password", "update_user"),
    ("x08", "distractor", "Em quais grupos o usuário u_77 está?", "list_user_groups", "list_groups"),
    ("x09", "distractor", "Lista os grupos da organização org_2.", "list_groups", "list_user_groups"),
    ("x10", "distractor", "Passa a titularidade do grupo g_9 pro usuário u_31.", "transfer_group_ownership", "set_group_permissions"),
    ("x11", "distractor", "Define quem pode postar no grupo g_9.", "set_group_permissions", "transfer_group_ownership"),
    ("x12", "distractor", "O u_301 está online agora?", "get_user_status", "get_user_profile"),
    ("x13", "distractor", "A notificação not_55 foi entregue?", "get_notification_status", "get_user_status"),
    ("x14", "distractor", "Muda as preferências de notificação do u_12 para só e-mail.", "update_notification_settings", "update_user"),
    ("x15", "distractor", "Cancela aquele aviso que eu tinha agendado, o not_88.", "cancel_scheduled_notification", "delete_notification"),
    ("x16", "distractor", "Apaga a notificação not_88 da lista do usuário.", "delete_notification", "cancel_scheduled_notification"),

    # ---------- no_tool (16) ----------
    ("n01", "no_tool", "Bom dia! Tudo certo por aí?", None, None),
    ("n02", "no_tool", "Você consegue me explicar o que é uma API REST?", None, None),
    ("n03", "no_tool", "Qual a diferença entre HTTP e HTTPS?", None, None),
    ("n04", "no_tool", "Obrigado, era só isso mesmo.", None, None),
    ("n05", "no_tool", "Me dá uma ideia de nome pro nosso novo projeto.", None, None),
    ("n06", "no_tool", "Quantos dias tem fevereiro em ano bissexto?", None, None),
    ("n07", "no_tool", "Escreve um resumo de duas linhas sobre trabalho remoto.", None, None),
    ("n08", "no_tool", "O que você acha de reunião toda segunda de manhã?", None, None),
    ("n09", "no_tool", "Traduz 'prazo de entrega' para o inglês.", None, None),
    ("n10", "no_tool", "Você é uma inteligência artificial?", None, None),
    ("n11", "no_tool", "Me explica o que faz um product owner.", None, None),
    ("n12", "no_tool", "Como eu calculo a média de três notas?", None, None),
    ("n13", "no_tool", "Qual a capital da Austrália?", None, None),
    ("n14", "no_tool", "Sugere uma pauta pra retrospectiva de sprint.", None, None),
    ("n15", "no_tool", "Corrige a gramática: 'nós vai enviar o relatorio amanha'.", None, None),
    ("n16", "no_tool", "Nada por enquanto, depois eu te chamo.", None, None),
]

QUERIES = [
    {
        "id": qid,
        "type": qtype,
        "text": text,
        # normalizado para tupla: gabarito é sempre um conjunto de nomes aceitáveis
        # (vazio = nenhuma ferramenta deve ser chamada)
        "expected": () if expected is None else (expected,) if isinstance(expected, str) else expected,
        "lure": lure,
        "reviewed_by": None,
    }
    for qid, qtype, text, expected, lure in _QUERIES
]

QUERY_TYPES = ("direct", "ambiguous", "distractor", "no_tool")


if __name__ == "__main__":
    from collections import Counter
    from corpus import TOOLS_BY_NAME

    assert len({q["id"] for q in QUERIES}) == len(QUERIES), "ids duplicados"

    counts = Counter(q["type"] for q in QUERIES)
    assert set(counts) == set(QUERY_TYPES), f"tipos inesperados: {set(counts) - set(QUERY_TYPES)}"
    assert len(set(counts.values())) == 1, f"dataset desbalanceado: {dict(counts)}"

    for q in QUERIES:
        for name in q["expected"]:
            assert name in TOOLS_BY_NAME, f"{q['id']}: ferramenta inexistente no corpus: {name}"
        if q["lure"]:
            assert q["lure"] in TOOLS_BY_NAME, f"{q['id']}: lure inexistente: {q['lure']}"
            assert q["lure"] not in q["expected"], f"{q['id']}: lure não pode estar no gabarito"
        assert bool(q["expected"]) == (q["type"] != "no_tool"), f"{q['id']}: gabarito incoerente com o tipo"

    pendentes = sum(1 for q in QUERIES if not q["reviewed_by"])
    print(f"queries.py: ok — {len(QUERIES)} queries, {dict(counts)}")
    if pendentes:
        print(f"  AVISO: {pendentes} queries sem segundo avaliador (reviewed_by=None)")
