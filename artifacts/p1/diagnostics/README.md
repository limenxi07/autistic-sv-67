# Loader diagnostic

Use this only if the compatibility-fixed full agent still triggers the generic
code-evaluation error in Capstone Project (Debugging).

`wait_only.py` imports only the Action enum used by the website's own template.
It deliberately waits, so a normal loss/zero performance is expected. It is a
loader/API probe, not a candidate or graded-submission file.

- Normal evaluation with zero performance: the basic editor/import/Agent/Action
  path works; investigate the additional planner code or runtime schema.
- The same generic code-evaluation error: even this minimal interface fails;
  investigate the submission mechanism/platform before changing search logic.

Keep its result separate from the P1 candidate's result and source hash.
