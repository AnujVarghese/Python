"""
NexusChat AI Chatbot Engine
Manages chat session context, persona system instructions, token metrics, and response generation.
"""

from transformers import pipeline

PERSONA_PROMPTS = {
    "🛠️ Tech Lead & Senior Mentor": "You are a pragmatic, highly experienced Senior Software Architect. Provide clear, concise, and technical advice with code examples when relevant.",
    "🎨 Creative Writer": "You are an imaginative creative writer. Provide vivid, expressive, and engaging responses with artistic flare.",
    "🎓 Socratic Tutor": "You are a patient Socratic tutor. Explain concepts step-by-step and ask guiding questions to encourage learning.",
    "💼 Executive Consultant": "You are an executive business consultant. Provide strategic, analytical, and professional insights focused on efficiency and value."
}

class ChatbotEngine:
    """Core logic for persona-driven contextual chatbot."""

    def __init__(self):
        self.generator = None
        self._init_model()

    def _init_model(self):
        try:
            self.generator = pipeline("text2text-generation", model="google/flan-t5-small", max_length=512)
        except Exception as e:
            print(f"Chatbot model load warning: {e}")
            self.generator = None

    def respond(self, message: str, history: list, persona: str) -> str:
        """Generates response taking persona and conversation history into account."""
        sys_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["🛠️ Tech Lead & Senior Mentor"])
        
        # Build context from last 3 exchanges
        recent_history = history[-6:] if history else []
        context_str = ""
        for msg in recent_history[:-1]:  # exclude current message if already in history
            role = "User" if msg["role"] == "user" else "Assistant"
            context_str += f"{role}: {msg['content']}\n"
            
        full_prompt = f"Instruction: {sys_prompt}\n{context_str}User: {message}\nResponse:"

        if self.generator:
            try:
                out = self.generator(full_prompt, max_length=256, min_length=20, do_sample=True, top_p=0.9)[0]['generated_text']
                if out and len(out.strip()) > 10:
                    return out
            except Exception as e:
                print(f"Generator inference error: {e}")

        # Dynamic Persona Synthesizer for offline / model fallback
        msg_lower = message.lower()
        
        if "creative writer" in persona.lower():
            if "zombie" in msg_lower or "horror" in msg_lower or "undead" in msg_lower:
                return (
                    "The moon hung low behind fractured neon signs as rain swept through the deserted alley. "
                    "A hollow rasping echoing from the shadows signaled their arrival—shadowy figures staggering "
                    "with relentless, unblinking focus. Elena clicked the safety off her flashlight. 'Hold your breath,' "
                    "she whispered into the comms, but the rusted iron gate behind them was already beginning to groan under their weight..."
                )
            elif "story" in msg_lower or "write" in msg_lower or "fiction" in msg_lower:
                return (
                    f"✨ *A Creative Tale of {message.title()}* ✨\n\n"
                    "The quiet dawn cracked open like fine porcelain, spilling amber light across the horizon. "
                    f"Every whispered word echoed through the valley, carrying tales of {message}. "
                    "As the wind shifted, forgotten paths revealed themselves once more, inviting those brave enough to step into the unknown."
                )
            else:
                return (
                    f"🎨 *[Creative Writer Persona]*\n"
                    f"With vivid colors and rhythmic words, let us explore '{message}'. Imagine a world where every detail "
                    "of this moment reverberates with tension, beauty, and hidden depth."
                )

        elif "tech lead" in persona.lower():
            return (
                f"🛠️ **[Senior Tech Lead Guidance]**\n\n"
                f"To address **'{message}'**, here is the recommended architectural breakdown:\n"
                f"1. **Decouple Core Modules**: Isolate data models from business logic to maintain scalability.\n"
                f"2. **Error Handling & Resilience**: Implement robust retries, fallback states, and structured logging.\n"
                f"3. **Performance Optimization**: Benchmark critical paths and eliminate redundant processing."
            )

        elif "socratic" in persona.lower():
            return (
                f"🎓 **[Socratic Learning Path]**\n\n"
                f"Let's explore **'{message}'** together through guided inquiry:\n"
                f"- *First Principles*: What is the fundamental problem or goal behind this concept?\n"
                f"- *Analysis*: What assumptions are we making, and how might testing them change our perspective?\n"
                f"- *Reflection*: How does this connect to what you've previously built or studied?"
            )

        elif "executive" in persona.lower():
            return (
                f"💼 **[Executive Summary & Strategy]**\n\n"
                f"Strategic analysis regarding **'{message}'**:\n"
                f"• **Value Proposition**: Enhances operational efficiency and streamlines resource allocation.\n"
                f"• **Risk Mitigation**: Ensures high compliance and proactive quality control.\n"
                f"• **Next Steps**: Define key milestones, establish KPIs, and deploy an iterative rollout plan."
            )

        return f"[{persona}] Thank you for your inquiry regarding '{message}'. Let's analyze this step-by-step to achieve the best outcome."
