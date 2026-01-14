import random

class NotificationService:
    def __init__(self):
        # Personality-based message templates
        self.templates = {
            "SWITCH_TO_CHATBOT": [
                "This concept is tricky! Want to talk it out with the AI tutor?",
                "You've been focused for a while. Let's try a quick chat session?",
                "Stuck? I can explain this in 3 simple steps in the chat."
            ],
            "SWITCH_TO_VIDEO": [
                "Visuals might help here. Want to watch a quick 2-minute breakdown?",
                "Let's switch gears! Check out this simulation video.",
            ],
            "ADJUST_SCHEDULE": [
                "You're ahead of schedule! I've updated your roadmap to give you a break.",
                "Looks like we need more time on this. I've adjusted your weekly goals."
            ]
        }

    async def craft_nudge(self, action_type: str, student_id: str):
        """Wraps the RL action with a human-friendly message."""
        messages = self.templates.get(action_type, ["How is your session going?"])
        return {
            "student_id": student_id,
            "message": random.choice(messages),
            "timestamp": "2026-01-14T10:40:11Z"
        }

notification_service = NotificationService()