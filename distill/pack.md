# Distilled behavior examples

Learned from 8 successful agent run(s). Imitate this reasoning style:

## Example 1 — `brain_read`

Situation: on about:blank, page shows: (no snapshot)

Thought: I'll start by reading the previous run's memory file and navigating to the site in parallel.

Action: `brain_read({"path": "workflows/site-watch/memory/last-run.md"})`

## Example 2 — `brain_write`

Situation: on https://example.com/, page shows: (no snapshot)

Thought: I have the content. The site is reachable, title is "Example Domain". There's no previous run memory file, so I can't compare against a prior state — I'll treat this as the baseline (first run). Let me record the current state to memory for the next run.

Action: `brain_write({"content": "# site-watch \u2014 last run\n\nRun timestamp: current scheduled run (6h interval)\n\n## Site: https://example.com\n- HTTP reachable: yes\n- Page title: Example Domain\n- Status: OK\n\n### Headline content (baseline)\nPrimary headline: \"This domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.\"\n\nFull visible text:\nThis domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.\n\n\u0647\u0630\u0627 \u0627\u0644\u0646\u0637\u0627\u0642 \u0645\u064f\u062e\u0635\u0635 \u0644\u0644\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0641\u064a \u0623\u0645\u062b\u0644\u0629 \u0627\u0644\u062a\u0648\u062b\u064a\u0642 \u062f\u0648\u0646 \u0627\u0644\u062d\u0627\u062c\u0629 \u0625\u0644\u0649 \u0625\u0630\u0646. \u0647\u0630\u0647 \u0644\u064a\u0633\u062a \u062e\u062f\u0645\u0629\u060c \u064a\u064f\u0631\u062c\u0649 \u062a\u062c\u0646\u0628 \u0627\u0644\u0627\u0639\u062a\u0645\u0627\u062f \u0639\u0644\u064a\u0647\u0627 \u0644\u0623\u063a\u0631\u0627\u0636 \u0627\u0644\u0627\u062e\u062a\u0628\u0627\u0631 \u0648\u0627\u0644\u0645\u0631\u0627\u0642\u0628\u0629.\n\n\u8be5\u57df\u540d\u4ec5\u7528\u4e8e\u6587\u6863\u793a\u4f8b\uff0c\u65e0\u9700\u83b7\u5f97\u8bb8\u53ef\u3002\u8fd9\u5e76\u975e\u4e00\u9879\u670d\u52a1\uff0c\u8bf7\u52ff\u5c06\u5176\u7528\u4e8e\u6d4b\u8bd5\u548c\u76d1\u63a7\u76ee\u7684\u3002\n\nL\u2019usage de ce domaine est r\u00e9serv\u00e9 \u00e0 des exemples de documentation, sans autorisation pr\u00e9alable. Il ne s\u2019agit pas d\u2019un service ; son utilisation \u00e0 des fins de test ou de surveillance est \u00e0 \u00e9viter.\n\n\u0414\u0430\u043d\u043d\u044b\u0439 \u0434\u043e\u043c\u0435\u043d \u043f\u0440\u0435\u0434\u043d\u0430\u0437\u043d\u0430\u0447\u0435\u043d \u0434\u043b\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u044f \u0432 \u043f\u0440\u0438\u043c\u0435\u0440\u0430\u0445 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u0438 \u0431\u0435\u0437 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u0438 \u043f\u043e\u043b\u0443\u0447\u0435\u043d\u0438\u044f \u043f\u0440\u0435\u0434\u0432\u0430\u0440\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0433\u043e \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u044f. \u042d\u0442\u043e \u043d\u0435 \u0441\u0435\u0440\u0432\u0438\u0441; \u043d\u0435 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f \u0435\u0433\u043e \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u0438\u0435 \u0434\u043b\u044f \u0442\u0435\u0441\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f \u0438 \u043c\u043e\u043d\u0438\u0442\u043e\u0440\u0438\u043d\u0433\u0430.\n\nEste dominio est\u00e1 destinado al uso en ejemplos de documentaci\u00f3n sin necesidad de permiso. Esto no es un servicio; evitar utilizarlo para realizar pruebas o monitoreos.\n\nLearn more\n", "path": "workflows/site-watch/memory/last-run.md"})`

## Example 3 — `browser_click`

Situation: on https://example.com/, page shows: URL: https://example.com/
Title: Example Domain
Interactive elements:
  [e1] a: Learn more

Thought: I have the page content. Now let me click the "Learn more" link to determine its destination URL.

Action: `browser_click({"ref": "e1"})`

## Example 4 — `browser_get_text`

Situation: on https://example.com/, page shows: (no snapshot)

Thought: The site loaded successfully. The page title is "Example Domain" and it's reachable. There is no previous memory file, so this appears to be the first run. Let me get the full text content to record it, and save the current state to memory.

Action: `browser_get_text({})`

## Example 5 — `browser_navigate`

Situation: on https://example.com/, page shows: (no snapshot)

Thought: The page loaded successfully with the title "Example Domain". I need to verify the exact `<h1>` text, since the visible text doesn't clearly show a headline. Let me check the page source to confirm.

Action: `browser_navigate({"url": "view-source:https://example.com/"})`

## Example 6 — `brain_write`

Situation: on https://lobste.rs/, page shows: (no snapshot)

Thought: The page loaded successfully. I can see the #1 story at the top of the front page. Let me record the findings to the brain.

Action: `brain_write({"content": "Title: I got targeted: Trying to get your credentials via a git post-checkout hook | Score: 86 | Comments: 29 | Timestamp: recorded live at observation of https://lobste.rs front page (story shown as \"authored by frankwiles 18 hours ago\")\n", "path": "lobsters/top_story.txt"})`
