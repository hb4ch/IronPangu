// Run after rendering execution.html. Requires Playwright (see docs/plan-viewer.md).
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(
  require.resolve("playwright", {
    paths: [path.resolve(".deploy/plan-view-test"), process.cwd()],
  }),
);
(async () => {
  const file = process.argv[2];
  if (!file)
    throw Error(
      "usage: node scripts/test-plan-viewer.cjs EXECUTION.html [SCREENSHOT_DIR]",
    );
  const browser = await chromium.launch({
    headless: true,
    executablePath:
      process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1600, height: 1050 },
    });
    const errors = [],
      requests = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("request", (r) => {
      if (/^https?:/.test(r.url())) requests.push(r.url());
    });
    await page.goto(pathToFileURL(path.resolve(file)).href);
    assert.equal(await page.locator(".layer-card").count(), 46);
    assert.equal(
      await page.locator(".node").count(),
      0,
      "overview must not render primitive nodes",
    );
    if (process.argv[3])
      await page.screenshot({
        path: path.join(process.argv[3], "plan-overview.png"),
      });
    await page.locator('[data-layer="hidden.i2.block"]').click();
    await page.waitForFunction(() => planInspector.scope === "hidden.i2.block");
    assert.equal(await page.locator(".node").count(), 13);
    await page.locator('.node[data-id="hidden.i2.block.attention"]').click();
    await page.locator("#expand-module").click();
    await page.waitForFunction(
      () => planInspector.scope === "hidden.i2.block.attention",
    );
    assert.equal(await page.locator(".node").count(), 19);
    const qa = await page.evaluate(() =>
      JSON.parse(
        document.querySelector("#plan-data").textContent,
      ).graph.nodes.findIndex((n) => n.name === "hidden.i2.block.attention.qa"),
    );
    await page.locator('.node[data-id="op:' + qa + '"]').click();
    assert.match(
      await page.locator("#inspector").innerText(),
      /tensor \[1,024\]/,
    );
    assert.match(
      await page.locator("#inspector").innerText(),
      /model\.layers\.2\.self_attn\.q_a_proj/,
    );
    assert.ok((await page.locator(".edge.highlight").count()) > 0);
    if (process.argv[3])
      await page.screenshot({
        path: path.join(process.argv[3], "plan-attention.png"),
      });
    await page.locator("#inspector [data-op]").first().click();
    await page.waitForFunction(() => planInspector.scope === "hidden.i2.block");
    assert.match(await page.locator("#inspector").innerText(), /attn_norm/);
    await page.goBack();
    await page.waitForFunction(
      () => planInspector.scope === "hidden.i2.block.attention",
    );
    await page.locator("#search").fill("q_a_proj");
    assert.equal(await page.locator(".search-result").count(), 46);
    await page.locator(".search-result").last().click();
    await page.waitForFunction(
      () => planInspector.scope === "hidden.i45.block.attention",
    );
    await page.locator("#search").fill("does-not-exist");
    assert.equal(await page.locator(".search-result").count(), 0);
    await page.locator("#search").fill("");
    const scopes = await page.evaluate(() =>
      [...planInspector.groups.keys()].filter((id) => id !== "root"),
    );
    let maxVisible = 0;
    for (const id of scopes) {
      await page.evaluate((id) => {
        location.hash = encodeURIComponent(id);
      }, id);
      await page.waitForFunction((id) => planInspector.scope === id, id);
      const stats = await page.evaluate(() => {
        const data = JSON.parse(
          document.querySelector("#plan-data").textContent,
        ).graph;
        const group = planInspector.groups.get(planInspector.scope);
        const owners = new Map(
          group.children.flatMap((g) => g.leaves.map((i) => [i, g.id])),
        );
        const edges = new Set(
          group.leaves.flatMap((i) =>
            data.nodes[i].inputs
              .filter((j) => owners.has(j) && owners.get(j) !== owners.get(i))
              .map((j) => owners.get(j) + ">" + owners.get(i)),
          ),
        );
        const drawn = new Set(
          [...document.querySelectorAll(".edge")].map(
            (e) => e.dataset.from + ">" + e.dataset.to,
          ),
        );
        return {
          expected: [...edges].sort(),
          drawn: [...drawn].sort(),
          count: planInspector.visibleCount,
        };
      });
      assert.deepEqual(
        stats.drawn,
        stats.expected,
        "aggregated dependencies: " + id,
      );
      maxVisible = Math.max(maxVisible, stats.count);
    }
    await page.evaluate(() => {
      location.hash = "root";
    });
    await page.waitForFunction(() => planInspector.scope === "root");
    await page.locator("#entry").click();
    await page.waitForFunction(() => planInspector.scope === "shell:Input");
    assert.equal(await page.locator(".node").count(), 2);
    await page.locator("#up").click();
    await page.waitForFunction(() => planInspector.scope === "root");
    await page.locator("#exit").click();
    await page.waitForFunction(() => planInspector.scope === "shell:Output");
    assert.equal(await page.locator(".node").count(), 4);
    await page.setViewportSize({ width: 760, height: 900 });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, [], "viewer must be offline");
    console.log(
      JSON.stringify({
        scopesChecked: scopes.length,
        maxVisible,
        browserErrors: errors.length,
        networkRequests: requests.length,
        navigation: "passed",
        validatedShapesAndBindings: "passed",
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
