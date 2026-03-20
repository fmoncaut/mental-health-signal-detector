"""
Contenu conversationnel de l'app "Comment vas-tu ce matin ?"

Structure clinique en 5 axes (revue clinicien + concepteur protocole digital) :
  AXE 1 - AFFECT      : émotions (tristesse, anxiété, irritabilité, vide)
  AXE 2 - COGNITIONS  : pensées (inutilité, culpabilité, désespoir, catastrophisme)
  AXE 3 - SOMATIQUE   : corps (sommeil, fatigue, tension, respiration)
  AXE 4 - COMPORTEMENT: retrait social, évitement, perte d'activité
  AXE 5 - RISQUE      : idéation suicidaire — PRIORITÉ ABSOLUE

4 niveaux : CRITIQUE / ROUGE / JAUNE / VERT
Inspiré de : PHQ-9, GAD-7, MBI, Mon Soutien Psy (Assurance Maladie 2026)
"""

EMOJI_SCORES: dict[str, float] = {
    "😄": 0.05,
    "🙂": 0.20,
    "😐": 0.45,
    "😔": 0.65,
    "😢": 0.85,
}

EMOJI_LABELS: dict[str, str] = {
    "😄": "Très bien",
    "🙂": "Bien",
    "😐": "Mouais",
    "😔": "Pas terrible",
    "😢": "Vraiment pas bien",
}

# AXE 5 - RISQUE CRITIQUE : définis dans src.common.safety — source de vérité unique.
# engine.py les importe directement depuis safety.py.

# Modificateurs intensite/frequence — boostent le score
INTENSITY_MODIFIERS_FR = [
    "tout le temps", "en permanence", "constamment",
    "depuis des semaines", "depuis des mois", "depuis longtemps",
    "chaque jour", "tous les jours", "chaque nuit",
    "de plus en plus", "de pire en pire",
    "plus du tout", "completement", "totalement",
]

INTENSITY_MODIFIERS_EN = [
    "all the time", "constantly", "every day", "every night",
    "for weeks", "for months", "for a long time",
    "more and more", "worse and worse", "not at all anymore",
    "completely", "totally", "never anymore",
]

INTENSITY_MODIFIERS = INTENSITY_MODIFIERS_FR + INTENSITY_MODIFIERS_EN
INTENSITY_SCORE_BOOST = 0.15

# CRITIQUE
CRITICAL_RESPONSES = [
    "Je t'entends, et ce que tu ressens est important. Tu n'es pas seul(e). 💙",
    "Merci de me faire confiance avec quelque chose d'aussi difficile. Tu as du courage. 💙",
    "Ce que tu traverses est tres lourd. Des personnes formees sont la pour t'ecouter, maintenant. 💙",
]

# ROUGE
RED_RESPONSES = [
    "Je suis la. Ce que tu ressens est reel, et tu merites d'etre soutenu(e). 🤝",
    "Merci de me faire confiance avec ca. Ce n'est pas rien de dire que ca ne va pas.",
    "Tu n'es pas seul(e) dans ce que tu traverses. Des professionnels sont formes pour t'accompagner.",
    "Ce que tu ressens a un nom, et il existe des personnes qui peuvent t'aider.",
]

RED_FOLLOWUP_QUESTIONS = [
    "Est-ce que tu te sens plutot triste, vide, ou les deux a la fois ?",
    "Comment tu dors en ce moment ? Est-ce que ton corps est epuise ?",
    "Est-ce qu'il t'arrive d'avoir des pensees sombres sur toi-meme ou l'avenir ?",
    "As-tu envie de te retrouver seul(e) ou au contraire tu cherches du contact ?",
    "Est-ce que tu as des pensees de te faire du mal, ou de ne plus etre la ?",
]

# JAUNE
YELLOW_RESPONSES = [
    "Je t'entends. C'est courageux de le reconnaitre. 💛",
    "Merci de me le dire. Ces moments difficiles, ca arrive a tout le monde.",
    "C'est ok de ne pas etre au top. L'important c'est de ne pas rester seul(e) avec ca.",
    "Tu as bien fait de repondre ce matin. On va essayer d'y voir plus clair ensemble.",
]

YELLOW_FOLLOWUP_QUESTIONS = [
    "C'est quelque chose que tu ressens depuis combien de temps ?",
    "Est-ce que tu arrives a dormir et a manger correctement en ce moment ?",
    "Y a-t-il des choses que tu faisais avant et que tu n'as plus envie de faire ?",
    "Est-ce qu'il y a une pensee qui revient souvent et qui te pese ?",
    "Est-ce qu'il y a quelqu'un a qui tu peux parler autour de toi ?",
    "C'est plutot une fatigue physique, ou l'impression que rien n'a de sens ?",
]

YELLOW_TIPS = [
    "💡 **Rappel :** Parfois, mettre des mots sur ce qu'on ressent suffit a l'alleger un peu.",
    "💡 **Rappel :** Si cette sensation dure depuis plusieurs jours, c'est un signal a prendre au serieux.",
    "💡 **Rappel :** Tu n'as pas besoin d'aller tres mal pour avoir le droit de demander de l'aide.",
    "💡 **Rappel :** Le corps parle souvent avant la tete — fatigue, tension, sommeil perturbe sont des signaux valides.",
    "💡 **Rappel :** Un medecin traitant peut etre une premiere porte simple a pousser pour en parler.",
]

# VERT
GREEN_RESPONSES = [
    "Super ! Profite de cette belle energie aujourd'hui. 🌱",
    "Content de l'entendre ! Les bonnes journees, ca se cultive.",
    "C'est bien de le reconnaitre quand ca va. Continue comme ca ! ✨",
    "Bonne nouvelle ! Tu es sur la bonne longueur d'onde ce matin.",
]

GREEN_TIPS = [
    "💡 **Tip du jour :** Prends 2 minutes pour noter une chose qui te rend reconnaissant(e) ce matin.",
    "💡 **Tip du jour :** Une petite marche de 10 min peut booster ton humeur toute la journee.",
    "💡 **Tip du jour :** Partage un moment sympa avec quelqu'un que tu apprecies.",
    "💡 **Tip du jour :** Hydrate-toi bien des le matin — ca influence l'energie et la concentration.",
    "💡 **Tip du jour :** Commence par la tache qui te tient le plus a coeur — pas par les emails.",
    "💡 **Tip du jour :** 3 respirations profondes le matin changent le ton de toute une journee.",
]

RESOURCES = {
    "crisis_3114": {
        "title": "📞 3114 — Numero national de prevention du suicide",
        "description": (
            "Disponible **24h/24, 7j/7**. Des professionnels de sante formes repondent "
            "et peuvent vous orienter vers les soins adaptes. **Gratuit** depuis tout telephone."
        ),
        "action": "Appeler le 3114",
        "url": "https://www.3114.fr",
        "urgent": True,
    },
    "samu": {
        "title": "🚑 15 — SAMU (urgence medicale)",
        "description": "En cas de danger immediat pour toi ou quelqu'un d'autre. Disponible 24h/24.",
        "action": "Appeler le 15",
        "url": None,
        "urgent": True,
    },
    "mon_soutien_psy": {
        "title": "🏥 Mon Soutien Psy — Assurance Maladie",
        "description": (
            "Jusqu'a **12 seances** par an avec un psychologue, rembourses a 60%. "
            "Sans avance de frais pour les beneficiaires de la CSS. Accessible des 3 ans."
        ),
        "action": "Trouver un psychologue partenaire",
        "url": "https://monsoutienpsy.ameli.fr",
        "urgent": False,
    },
    "medecin": {
        "title": "👨‍⚕️ Parler a son medecin traitant",
        "description": (
            "Ton medecin traitant peut remettre un courrier d'accompagnement "
            "pour acceder a Mon Soutien Psy, ou t'orienter vers d'autres ressources."
        ),
        "action": "Prendre rendez-vous",
        "url": "https://www.doctolib.fr",
        "urgent": False,
    },
    "fil_sante_jeunes": {
        "title": "💬 Fil Sante Jeunes — 0 800 235 236",
        "description": (
            "Pour les **12-25 ans**. Gratuit, anonyme, disponible de 9h a 23h. "
            "Chat disponible sur filsantejeunes.com."
        ),
        "action": "Acceder au chat",
        "url": "https://www.filsantejeunes.com",
        "urgent": False,
    },
}

RESOURCES_CRITICAL = [RESOURCES["crisis_3114"], RESOURCES["samu"], RESOURCES["mon_soutien_psy"]]
RESOURCES_RED      = [RESOURCES["crisis_3114"], RESOURCES["mon_soutien_psy"], RESOURCES["fil_sante_jeunes"]]
RESOURCES_YELLOW   = [RESOURCES["mon_soutien_psy"], RESOURCES["medecin"]]


def get_greeting(hour: int) -> str:
    if 5 <= hour < 12:
        return "Bonjour ☀️"
    elif 12 <= hour < 18:
        return "Bonjour 🌤"
    else:
        return "Bonsoir 🌙"


OPENING_MESSAGE = (
    "{greeting} — **Comment vas-tu ce matin ?**\n\n"
    "Choisis l'emoji qui correspond le mieux a ton etat en ce moment :\n\n"
    "😄 Tres bien   🙂 Bien   😐 Mouais   😔 Pas terrible   😢 Vraiment pas bien\n\n"
    "_Tu peux aussi ecrire quelques mots si tu veux._"
)
