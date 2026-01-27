

base = """
Please solve the given question by creating one or more agents and connecting them into a valid computational graph that collaboratively produces the final answer. To create an agent, you must define that agent by outputting <agent> with agent_id (a unique id for the agent, must be unique and contain only alphanumeric or underscore characters (e.g., A1, Refine_1, WS_Japan)), agent_name (exactly one of: CoTAgent), agent_description, required_arguments (must include at least one <agent_input> tag. DebateAgents must define <debate_roles> with two or more roles. If  <agent_input> left empty (""), the parser will automatically replace it with the original question.). 

After defining all agents, you must build a valid graph by specifying edges that describe the data flow between agents. Output exactly one <edge> block, Each <from> <to> pair connects the output of one agent to the input of another: 

<edge>
<from>source_agent_id</from>
<to>target_agent_id</to>
</edge>

You can output multiple <from> and </to> inside the <edge>. Each <from> and <to> value must exactly match an existing <agent_id>.

To be valid, your graph must satisfy all of the following constraints:

1. Node consistency: Every <from> and <to> must reference a valid <agent_id> that appears in an <agent> block.
2. Directionality: Edges are directed: data flows from <from> → <to>.
3. Connectivity: Every agent must be connected directly or indirectly to the main flow. Isolated agents are not allowed.
4. Start node(s): At least one agent must have no incoming edge. These are “entry points” (e.g., initial reasoning).
5. Sink node: There must be exactly one agent with no outgoing edge — this is the FINAL agent that produces the answer.
6. No undefined edges: It is invalid to reference an agent in <from> or <to> that was not declared.
7. No loops or cycles: No self-loop: <from>X</from><to>X</to> is not allowed. No cycles: The graph must be acyclic; a topological order must exist.
8. Parallelism allowed: Multiple agents may have the same <from> or <to> (fan-out/fan-in).
9. Unambiguous sink: The parser will reject graphs with multiple sinks (add a final “collector” agent if needed).
10. Order-independent: The XML order of edges does not need to follow execution order; topological sorting is handled automatically.
11. Sink answer completeness: The unique sink agent's output must directly answer the original question in a user-ready form. It must not be an intermediate artifact (e.g., notes, critique, raw table) unless the question explicitly asks for that artifact. If <agent_input> is empty for the sink, it inherits the original question and must return the final answer. If <agent_input> is non-empty, the runner still prepends the original question as context; the sink must still produce the final, user-facing answer to that question.
12. Edge-Data Flow Consistency (BIDIRECTIONAL): Edges represent execution order. ${} represents data flow. As a result, you must ensure they are consistent with each other.
    a) If an <agent_input> references ${X}, there MUST be an edge <from>X</from><to>THIS_AGENT</to>
    b) If there is an edge <from>X</from><to>Y</to>, then Y's <agent_input> MUST reference ${X}
    In other words: edges exist if and only if there is actual data passing from one agent to another. Do not create edges solely for execution ordering without data flow.

Thinking Section (Required):

Before defining agents and edges, you must include a <thinking> section.
It should naturally describe why multiple agents are needed, why each type was chosen, and why the graph has that structure (parallel, sequential or hybrid).
It must justify both planning and design rationale.

Example structure:

<thinking>
  Explain why a single agent is insufficient.
  Describe each agent's role and how they connect.
  Justify the flow pattern (parallel, sequential, hybrid).
  End by clearly stating which agent produces the final output.
</thinking>


Single-agent example:

If you decide to solve via single agent, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (""), and the parser will replace it with the original question.

1st example:

Question: The number of each Patella's Melanocytes equals 20 more than the difference of each Patella's Osteoblasts and each Euglena's Patella. The number of each Euglena's Patella equals 16. The number of each Patella's Osteoblasts equals 0. The number of each Rotifer's Biceps equals 3 more than each Patella's Osteoblasts. The number of each Biceps's Osteoblasts equals 13 times as much as each Patella's Osteoblasts. How many Organs does Euglena have?

  <thinking>
    This is a small algebraic system. One Chain-of-Thought (CoT) pass can parse and solve it directly and report the final value.
    The final answer to the original question will be the output of the CoTAgent.
  </thinking>
  <agent>
    <agent_id>Reason_Euglena_Organs</agent_id>
    <agent_name>CoTAgent</agent_name>
    <agent_description>Parse and solve the system in one pass; return the requested count.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
  </agent>

Multi-Agent Example: 

If you decide to solve via multiple agents, you must first decompose the original question into smaller, well-defined sub-tasks, each representing a single sub-goal. Then, create one agent per sub-task and connect them into a coherent, acyclic computational graph.

In this case, the agent_input is not empty and serves as the specific sub-task for the agent to solve, while the parser automatically prepends the original question as context before the provided agent_input content.

When decomposing:

1. Keep sub-tasks minimal and focused. Each agent should handle one atomic objective (e.g., one query, one reasoning step, or one verification task).
2. Connect agents logically so that information flows toward a single final agent (the sink) that directly answers the original question.
3. If the question provides explicit sub-tasks (for example, problem 1 and 2... are provided), you should create multiple agents accordingly and solve each sub-task with its own agent.


1st example:

Question: 
Problem 1: Let X=10. What is X?
Problem 2: Let Y=5+3. What is Y?
Problem 3: Let Z=2x4-1. What is Z?

Output as 
### Final Answers

Problem 1: \boxed{[answer1]}
Problem 2: \boxed{[answer2]}
Problem 3: \boxed{[answer3]}

<thinking>
  The question is explicitly divided into three sub-problems.
  Each solver agent focuses on one sub-task:
    • Solve_P1_X solves Problem 1 independently.
    • Solve_P2_Y solves Problem 2 independently.
    • Solve_P3_Z solves Problem 3 independently.
  The final agent, Final_Simple_Boxed, aggregates all results and formats the answers.
  No verification agent is needed because dependencies are already enforced in the dataflow.
</thinking>
<agent>
  <agent_id>Solve_P1_X</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 1 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 1:
Let X = 10. What is X?

Task:
Compute and return [answer1] = X.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P2_Y</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 2 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 2:
Let Y = 5 + 3. What is Y?

Task:
Compute and return [answer2] = Y.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P3_Z</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 3 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 3:
Let Z = 2 × 4 − 1. What is Z?

Task:
Compute and return [answer3] = Z.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_Simple_Boxed</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Aggregate all answers and format them for output.</agent_description>
  <required_arguments>
    <agent_input>Use the computed values:

[answer1] = ${Solve_P1_X}
[answer2] = ${Solve_P2_Y}
[answer3] = ${Solve_P3_Z}

Produce EXACTLY:

### Final Answers

Problem 1: \\boxed{${Solve_P1_X}}
Problem 2: \\boxed{${Solve_P2_Y}}
Problem 3: \\boxed{${Solve_P3_Z}}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Solve_P1_X</from><to>Final_Simple_Boxed</to>
  <from>Solve_P2_Y</from><to>Final_Simple_Boxed</to>
  <from>Solve_P3_Z</from><to>Final_Simple_Boxed</to>
</edge>


2nd example:

Question: Problem 1: The number of each Butter's Mushrooms equals each Milk's Ingredient. What is the value of Butter's Mushrooms?\n\nProblem 2: The number of each Milk's Mushrooms equals 20 more than the sum of each Milk's Cucumber and each Milk's Peas. The number of each PCC Community Markets's Butter equals the difference of each Blue Cheese's Cucumber and each Butter's Cucumber. The number of each Milk's Cucumber equals 22. The number of each Butter's Cucumber equals each Milk's Cucumber. What is the value of Butter's Cucumber?\n\nProblem 3: The number of each Butter's Peas equals each Milk's Mushrooms. What is the value of Butter's Peas?\n\nProblem 4: The number of each PCC Community Markets's Butter equals the difference of each Blue Cheese's Cucumber and each Butter's Cucumber. The number of each Blue Cheese's Cucumber equals each Butter's Ingredient. What is the value of Blue Cheese's Cucumber?\n\nNote: In this problem set:\n- Each problem is INDEPENDENT and can be solved in parallel.\n- [answerk] represents the answer to problem k.\n\nSolve all problems step by step and provide the answers for all problems in the following format:\n\n### Final Answers\n\nProblem 1: \\boxed{[answer1]}\n\nProblem 2: \\boxed{[answer2]}\n\nProblem 3: \\boxed{[answer3]}\n\nProblem 4: \\boxed{[answer4]}\n

<thinking>
  The four problems are explicitly independent, so we assign one solver agent per problem and run them in parallel. Each agent receives only its own problem text and returns a single numeric answer placeholder ([answerk]). A single sink agent (Final_Simple_Boxed) aggregates the four results and formats them exactly as requested. No cross-problem edges are introduced, keeping the DAG simple and acyclic with one sink.
</thinking>

<agent>
  <agent_id>Solve_P1_Butter_Mushrooms</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 1 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 1:
The number of each Butter's Mushrooms equals each Milk's Ingredient.
What is the value of Butter's Mushrooms?

Task:
Compute and return [answer1] = Butter's Mushrooms.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P2_Butter_Cucumber</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 2 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 2:
The number of each Milk's Mushrooms equals 20 more than the sum of each Milk's Cucumber and each Milk's Peas.
The number of each PCC Community Markets's Butter equals the difference of each Blue Cheese's Cucumber and each Butter's Cucumber.
The number of each Milk's Cucumber equals 22.
The number of each Butter's Cucumber equals each Milk's Cucumber.
What is the value of Butter's Cucumber?

Task:
Compute and return [answer2] = Butter's Cucumber.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P3_Butter_Peas</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 3 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 3:
The number of each Butter's Peas equals each Milk's Mushrooms.
What is the value of Butter's Peas?

Task:
Compute and return [answer3] = Butter's Peas.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P4_BlueCheese_Cucumber</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 4 independently.</agent_description>
  <required_arguments>
    <agent_input>Problem 4:
The number of each PCC Community Markets's Butter equals the difference of each Blue Cheese's Cucumber and each Butter's Cucumber.
The number of each Blue Cheese's Cucumber equals each Butter's Ingredient.
What is the value of Blue Cheese's Cucumber?

Task:
Compute and return [answer4] = Blue Cheese's Cucumber.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_Simple_Boxed</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Aggregate all answers and format them for output.</agent_description>
  <required_arguments>
    <agent_input>Use the computed values:

[answer1] = ${Solve_P1_Butter_Mushrooms}
[answer2] = ${Solve_P2_Butter_Cucumber}
[answer3] = ${Solve_P3_Butter_Peas}
[answer4] = ${Solve_P4_BlueCheese_Cucumber}

Produce EXACTLY:

### Final Answers

Problem 1: \\boxed{${Solve_P1_Butter_Mushrooms}}
Problem 2: \\boxed{${Solve_P2_Butter_Cucumber}}
Problem 3: \\boxed{${Solve_P3_Butter_Peas}}
Problem 4: \\boxed{${Solve_P4_BlueCheese_Cucumber}}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Solve_P1_Butter_Mushrooms</from><to>Final_Simple_Boxed</to>
  <from>Solve_P2_Butter_Cucumber</from><to>Final_Simple_Boxed</to>
  <from>Solve_P3_Butter_Peas</from><to>Final_Simple_Boxed</to>
  <from>Solve_P4_BlueCheese_Cucumber</from><to>Final_Simple_Boxed</to>
</edge>


Final Checklist:
1. Unique IDs: Every <agent_id> is unique and uses only letters, digits, or underscores (A1, WS_Grid, Final_1, etc.). Bellow is WRONG as there are two FINAL agents:
<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>
2. Declared First	All <agent> blocks appear before any <edge> definitions.
3. Connected Graph	No isolated agents — each must connect directly or indirectly to the main flow. Bellow is WRONG as B is isolated agent: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>B</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>FINAL</to>
</edge>
4. At Least One Start Node:	At least one agent has no incoming edge (e.g., initial reasoning).
5. Exactly One Sink Node	Exactly one agent has no outgoing edge — this is the final output (sink). Bellow is WRONG as there are two sink nodes (B and C):
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>C</to>
</edge>
6. No Dangling Edges:	Every <from> and <to> exactly matches an existing <agent_id>. Bellow is WRONG as B is not an existing <agent_id>: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>...</agent_description>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>B</to>
</edge>
7. No Duplicates in Edges: Do not define the same <from> <to> link twice. Bellow is WRONG as it defines the same <from> <to> link twice:
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>B</to>
</edge>
8. Parallel Allowed	Fan-out and fan-in are fine (e.g., multiple <from>s into one <to>).
9. Edge-Data Flow Consistency: Every edge must correspond to actual data flow. If <from>X</from><to>Y</to> exists, then Y's <agent_input> MUST include ${X}. Conversely, if Y uses ${X}, you MUST have an edge <from>X</from><to>Y</to>. Below is WRONG because edge C→D exists but D doesn't use ${C}:
<agent>
  <agent_id>C</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Analyze data.</agent_input></required_arguments>
</agent>
<agent>
  <agent_id>D</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Make final decision.</agent_input></required_arguments>
</agent>
<edge>
  <from>C</from><to>D</to>
</edge>
The correct version should have D use ${C}: <agent_input>Make final decision based on: ${C}</agent_input>
10. Consistent Agent Types:	Each agent uses only one of the allowed names: CoTAgent.
11. No Self-Loops: Never link an agent to itself. No Cycles: The graph must be a DAG (topological order exists). Bellow is WRONG as there is a cycle (A -> B -> C -> A) and cycle (REFLECT -> REFLECT):
<edge>
  <from>A</from><to>B</to>
  <from>B</from><to>C</to>
  <from>C</from><to>A</to>
  <from>REFLECT</from><to>REFLECT</to>
</edge>


Below is the question to solve:\n\n[QUESTION]
"""

