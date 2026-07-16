import logging
import functools

try:
    from autogen import ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

from risklayer.core.trajectory import active_trace_id
from risklayer.evaluators.hallucination import hallucination_guard
from risklayer.evaluators.security import security_guard

logger = logging.getLogger("risklayer.integrations.autogen")

def patch_autogen(swarm_trace_id: str = "autogen-swarm"):
    """
    Hooks into Microsoft AutoGen's ConversableAgent.
    Every time an agent generates a reply, it is routed through the risklayer Security
    and Hallucination cascading guards to ensure safety in multi-agent swarms.
    """
    if not AUTOGEN_AVAILABLE:
        logger.warning("AutoGen not installed. Cannot patch ConversableAgent.")
        return
        
    original_generate_reply = ConversableAgent.generate_reply
    
    @functools.wraps(original_generate_reply)
    def guarded_generate_reply(self, messages=None, sender=None, **kwargs):
        # Set the active trace ID so that multi-agent interactions are grouped in the DAG
        active_trace_id.set(f"{swarm_trace_id}_{self.name}")
        
        # We apply the security guard on the incoming messages (prevent prompt injection between agents)
        input_text = " ".join([m.get("content", "") for m in (messages or [])])
        
        @security_guard()
        def _check_incoming_messages():
            return True
            
        try:
            _check_incoming_messages(input_text)
        except Exception as e:
            logger.error(f"Security intercept on agent {self.name}: {str(e)}")
            return True, f"[BLOCKED BY risklayer] Intrusion attempt detected from {sender.name if sender else 'user'}."
            
        # Execute the actual reply generation
        is_final, reply = original_generate_reply(self, messages, sender, **kwargs)
        
        # We apply hallucination guard on the outgoing reply
        @hallucination_guard()
        def _check_outgoing_reply():
            return True
            
        try:
            # We mock the prompt as input_text and generated_text as the reply
            hallucination_guard.adaptive_score_fn(input_text, str(reply))
        except Exception as e:
            logger.error(f"Hallucination intercept on agent {self.name}: {str(e)}")
            return True, "[BLOCKED BY risklayer] Generated reply failed factual density bounds."
            
        return is_final, reply
        
    ConversableAgent.generate_reply = guarded_generate_reply
    logger.info("AutoGen ConversableAgent successfully instrumented for Multi-Agent swarms.")

def unpatch_autogen():
    if not AUTOGEN_AVAILABLE:
        return
    import importlib
    import autogen
    importlib.reload(autogen)
    logger.info("AutoGen restored to original state.")
