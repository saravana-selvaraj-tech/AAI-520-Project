from agent.graph import AgentGraph
from agent.state import EvidenceRecord
from planning.planner import Planner, TaskType
from tools.providers import InMemoryResearchProvider


def test_planner_creates_research_dimensions_and_terminal_tasks():
    tasks = Planner().plan("Analyze Microsoft's earnings, news, SEC filings, and risks")
    task_types = [task.task_type for task in tasks]
    assert task_types[:4] == [TaskType.FINANCIAL, TaskType.NEWS, TaskType.SEC, TaskType.RISK]
    assert task_types[-2:] == [TaskType.REPORT, TaskType.EVALUATION]


def test_agent_graph_preserves_provenance_and_creates_cohort3_answer():
    evidence = EvidenceRecord(
        source="SEC EDGAR",
        title="FY2025 annual report",
        content="Revenue increased 8% year over year.",
        ticker="AAPL",
        document_id="10k-2025",
        url="https://example.test/10k",
    )
    graph = AgentGraph(InMemoryResearchProvider(
        financial_data={"revenue": "8% growth"},
        filings=[evidence],
    ))

    state = graph.run("Analyze AAPL earnings and SEC filings")

    assert state.ticker == "AAPL"
    assert state.completed_tasks
    assert state.evidence[0].document_id == "10k-2025"
    assert "FY2025 annual report (SEC EDGAR)" in state.draft_report
    assert "revenue: 8% growth" in state.metadata["context"]


def test_router_reports_missing_evidence_without_inventing_claims():
    state = AgentGraph(InMemoryResearchProvider()).run("Analyze MSFT")
    assert "No retrieved evidence" in state.draft_report