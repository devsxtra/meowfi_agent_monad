from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task

from src.meowfi.tools.custom_tool import GetPoolData


@CrewBase
class MeowFiAgent():
    """Agent Class"""
    agents_config = 'config/agents.yaml'

    tasks_config = 'config/tasks.yaml'

    llm = LLM(model="gpt-4o", temperature=0.15)

    @agent
    def defi_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['defi_agent'],
            llm=self.llm,
            verbose=True
        )

    @agent
    def aggregator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['defi_agent'],
            llm=self.llm,
            tools=[GetPoolData()],
            verbose=True
        )

    @agent
    def strategist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['defi_analyst'],
            llm=LLM(model="gpt-4o", temperature=0.2),
            verbose=True
        )

    @task
    def parse_user_input(self) -> Task:
        return Task(
            config=self.tasks_config['parse_user_input']
        )

    @task
    def data_aggregator(self) -> Task:
        return Task(
            config=self.tasks_config['data_aggregator']
        )

    @task
    def lp_pool_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['lp_pool_analysis_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.defi_agent(), self.strategist_agent(), self.aggregator_agent()],
            tasks=[
                self.parse_user_input(), self.data_aggregator(), self.lp_pool_analysis_task()],
            process=Process.sequential,
            memory=True,
            verbose=True
        )
