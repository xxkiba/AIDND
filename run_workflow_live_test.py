# run_workflow_live_test.py
from agent_workflow import answer_query

if __name__ == "__main__":
    # 试一个会触发 look -> search -> fetch 的问题
    # 你也可以试法术/装备：如 "Show me details of 'Studded Leather Armor' (prefer 2014 SRD)."
    q = "Tell me about something called Merrow."
    print("USER:", q)
    out = answer_query(q, max_tool_steps=4)
    print("\nASSISTANT:", out)
