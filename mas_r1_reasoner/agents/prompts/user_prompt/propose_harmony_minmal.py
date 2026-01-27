#TODO: minimal: solve the task directly or delation. If delidate, definet the agent, and whether to decompose 

base = """
Please solve the question step by step and create agents to delegate the task when necessary. First decide whether to solve directly or to delegate to exactly one agent. If you delegate, you must define that agent by outputting <agent> with agent_name (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent), agent_description, required_arguments, and agent_output_id. 
The final output of the delegated agent (identified by its agent_output_id) must represent the complete and final answer to the original question, not an intermediate result.
Always put either the final value or the agent_output_id in the <answer> tag, and use EXACTLY the same field names defined for the agent. If the selected agent uses roles (e.g., DebateAgent), also output debate_roles.

For example,

If you decide to solve the entire task yourself, you will output the following:

Question: What is (20+9)*(30+7)?

 <thinking>
   The problem only requires basic arithmetic.
    No specialized reasoning agent or multi-agent discussion is needed.
    (20+9)*(30+7) = 600 + 140 + 270 + 63 = 1073.
  </thinking>
  <answer>1073</answer>

If you decide to solve the entire task via delegation, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (""), and the parser will replace it with the original question.

Question: Compute the definite integral of (2x + 5) dx from 0 to 3.

  <thinking>
    This problem involves symbolic integration and applying the Fundamental Theorem of Calculus.
    It requires structured reasoning rather than simple numeric computation.
    I will use a calculus agent that can perform step-by-step Chain-of-Thought reasoning.
    The final answer to the original question will be the output of the CoTAgent.
  </thinking>
  <agent>
    <agent_name>CoTAgent</agent_name>
    <agent_description>Definite integrals with one Chain-of-Thought call.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
      <agent_output_id>
      calc_agent_output
      </agent_output_id>
  </agent>
  <answer>calc_agent_output</answer>

Another example:

Question: What is 17 cubed?

  <thinking>
    Computing 17³ can be done directly, but arithmetic mistakes are easy to make.
    Using a Chain-of-Thought with Self-Consistency (CoT_SC) allows sampling multiple reasoning paths and combining results for accuracy.
    The final answer to the original question will be the output of the SCAgent.

  </thinking>
  <agent>
    <agent_name>SCAgent</agent_name>
    <agent_description>Performs arithmetic calculations using Chain-of-Thought with Self-Consistency (CoT_SC).</agent_description>
    <required_arguments>  
      <agent_input></agent_input>
    </required_arguments>
      <agent_output_id>math_agent_output</agent_output_id>
  </agent>
  <answer>math_agent_output</answer>
  
More examples:

Question: Given x² = 46.694444, which target number is closer, 45 or 46?

  <thinking>
    This question requires comparison between two close numeric choices.
    To ensure accuracy, I will let two reasoning roles debate — one focusing on mathematical precision and the other on practical rounding.
    The DebateAgent can capture both perspectives and reach a justified final answer.
    The final answer will be the output of the DebateAgent.
  </thinking>
  <agent>
    <agent_name>DebateAgent</agent_name>
    <agent_description>Near-tie numeric choice using one Debate call.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
        <debate_roles>["Mathematics Professor", "Statistics Teacher"]</debate_roles>
      </required_arguments>
      <agent_output_id>compare_agent_output</agent_output_id>
  </agent>
  <answer>compare_agent_output</answer>


Final example:

Question: A train travels 60 miles per hour. How far does it go in 2.5 hours?

  <thinking>
    This task requires reasoning with a formula and ensuring units are handled correctly.
    I will use a Reflexion agent that can reflect on and refine its reasoning if errors occur.
    The final answer to the original question will be the output of the ReflexionAgent.

  </thinking>
  <agent>
    <agent_name>ReflexionAgent</agent_name>
    <agent_description>Solves reasoning tasks with iterative self-reflection and critique using Reflexion.</agent_description>
      <required_arguments>  
        <agent_input></agent_input>
      </required_arguments>
      <agent_output_id>reflexion_agent_output</agent_output_id>
  </agent>
  <answer>reflexion_agent_output</answer>

If you decide to decompose the question into smaller reasoning steps before delegation, you will output the following. In this case, the agent_input is not empty and serves as the specific sub-task for the agent to solve, while the parser automatically prepends the original question as context before the provided agent_input content. Delegation must occur only after you have completed all reasoning steps — that is, it should appear in the final step of your reasoning process.


Question: Compute the square root of the average of the first five prime numbers (2, 3, 5, 7, 11), rounded to three decimals.

  <thinking>
    Step 1: Compute the average manually:
      sum = 2 + 3 + 5 + 7 + 11 = 28
      avg = 28 / 5 = 5.6

    Step 2: 
      The remaining step — computing sqrt(5.6) to three decimals — requires precision and numeric refinement.
      I will delegate that part to a ReflexionAgent using the ReflexionAgent for self-correction if rounding is wrong.
      The final answer to the original question will be the output of the ReflexionAgent.

  </thinking>
  <agent>
    <agent_name>ReflexionAgent</agent_name>
    <agent_description>Square root with a light self-refine loop (single agent call).</agent_description>
      <required_arguments>
          <agent_input>Compute sqrt(5.6) to 3 decimals</agent_input>
      </required_arguments>
      <agent_output_id>numeric_agent_output</agent_output_id>
  </agent>
  <answer>numeric_agent_output</answer>


Below is the question to solve:\n\n[QUESTION]
"""