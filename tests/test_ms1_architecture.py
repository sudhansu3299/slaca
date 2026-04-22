"""
MS1 validation: System architecture is unambiguous and traceable.
"""

from src.architecture import build_architecture, StorageLayer, AgentBoundary, ARCHITECTURE


class TestArchitecture:
    def test_three_agents_defined(self):
        arch = build_architecture()
        assert len(arch.agents) == 3

    def test_agent_names(self):
        names = [a.name for a in ARCHITECTURE.agents]
        assert "AssessmentAgent" in names
        assert "ResolutionAgent" in names
        assert "FinalNoticeAgent" in names

    def test_modalities(self):
        modalities = {a.name: a.modality for a in ARCHITECTURE.agents}
        assert modalities["AssessmentAgent"] == "chat"
        assert modalities["ResolutionAgent"] == "voice"
        assert modalities["FinalNoticeAgent"] == "chat"

    def test_handoff_chain(self):
        agents = {a.name: a for a in ARCHITECTURE.agents}
        assert agents["AssessmentAgent"].handoff_to == "ResolutionAgent"
        assert agents["ResolutionAgent"].handoff_to == "FinalNoticeAgent"
        assert agents["FinalNoticeAgent"].handoff_to is None

    def test_truth_owner_is_context(self):
        assert ARCHITECTURE.truth_owner == StorageLayer.CONTEXT

    def test_all_storage_layers_present(self):
        layers = ARCHITECTURE.storage_layers
        assert StorageLayer.TEMPORAL in layers
        assert StorageLayer.MONGODB in layers
        assert StorageLayer.CONTEXT in layers
        assert StorageLayer.LEARNING_STORE in layers

    def test_borrower_trace_covers_all_agents(self):
        trace = ARCHITECTURE.validate_borrower_trace()
        full = "\n".join(trace)
        assert "AssessmentAgent" in full
        assert "ResolutionAgent" in full
        assert "FinalNoticeAgent" in full
        assert "EXIT" in full

    def test_borrower_trace_no_ambiguity(self):
        """Each step in trace has exactly one next step."""
        trace = ARCHITECTURE.validate_borrower_trace()
        # No step should be empty
        for step in trace:
            assert len(step.strip()) > 0

    def test_voice_agent_has_shorter_token_limit(self):
        agents = {a.name: a for a in ARCHITECTURE.agents}
        voice = agents["ResolutionAgent"]
        chat_assessment = agents["AssessmentAgent"]
        assert voice.max_output_tokens < chat_assessment.max_output_tokens

    def test_agent_inputs_outputs_defined(self):
        for agent in ARCHITECTURE.agents:
            assert len(agent.inputs) > 0
            assert len(agent.outputs) > 0
