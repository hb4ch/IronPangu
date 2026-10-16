// Generic browser verification; accepts any artifact from view_token_plan.
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const { pathToFileURL } = require("node:url");
const { chromium } = require(
  require.resolve("playwright", {
    paths: [path.resolve(".deploy/plan-view-test"), process.cwd()],
  }),
);
(async () => {
  let file = process.argv[2];
  const screenshotDir = process.argv.slice(3).find((arg) => arg !== "--flat");
  if (!file)
    throw Error(
      "usage: node scripts/test-plan-viewer.cjs EXECUTION.html [SCREENSHOT_DIR] [--flat]",
    );
  if (process.argv.includes("--flat")) {
    const html = fs.readFileSync(file, "utf8");
    const pattern =
      /(<script type="application\/json" id="plan-data">)([\s\S]*?)(<\/script>)/;
    const match = html.match(pattern);
    assert.ok(match, "embedded plan payload");
    const data = JSON.parse(match[2]);
    data.graph.name = "FlatMetadataFixture";
    data.graph.nodes.forEach((node, i) => {
      delete node.source;
      node.name = "operation_" + i;
    });
    const payload = JSON.stringify(data)
      .replace(/</g, "\\u003c")
      .replace(/>/g, "\\u003e")
      .replace(/&/g, "\\u0026");
    fs.mkdirSync(".deploy", { recursive: true });
    file = ".deploy/viewer-flat-fixture.html";
    fs.writeFileSync(
      file,
      html.replace(pattern, (_, begin, _data, end) => begin + payload + end),
    );
  }
  const browser = await chromium.launch({
    headless: true,
    executablePath:
      process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
  });
  try {
    const page = await browser.newPage({
        viewport: { width: 1600, height: 1050 },
      }),
      errors = [],
      requests = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("request", (r) => {
      if (/^https?:/.test(r.url())) requests.push(r.url());
    });
    await page.goto(pathToFileURL(path.resolve(file)).href);
    assert.equal(await page.locator(".node").count(), 0);
    const initial = await page.evaluate(() => ({
      count: planInspector.groups.get("root").children.length,
      pageSize: planInspector.pageSize,
    }));
    assert.equal(
      await page.locator(".scope-card").count(),
      Math.min(initial.count, initial.pageSize),
    );
    const provenance = await page.evaluate(() => {
      const data = JSON.parse(
        document.querySelector("#plan-data").textContent,
      ).graph;
      const actual = [];
      for (const group of planInspector.groups.values())
        for (const child of group.children)
          if (child.index !== undefined) {
            const frames = [];
            for (let p = child.parent; p?.frame; p = p.parent)
              frames.unshift(p.frame);
            actual[child.index] = frames;
          }
      return {
        expected: data.nodes.map((n) => n.source?.frames || []),
        actual,
      };
    });
    assert.deepEqual(
      provenance.actual,
      provenance.expected,
      "preserve every source frame, regardless of module name",
    );
    if (screenshotDir)
      await page.screenshot({
        path: path.join(screenshotDir, "plan-overview.png"),
      });
    // Exercise visible controls, rather than relying only on direct scope URLs.
    await page.locator(".scope-card").first().click();
    await page.waitForFunction(() => planInspector.visibleCount > 0);
    if (initial.count > initial.pageSize) {
      await page.locator("#crumbs button").first().click();
      await page.waitForFunction(() => document.querySelector(".scope-card"));
      await page.locator("#next-page").click();
      await page.waitForFunction(() => planInspector.pageIndex === 1);
      await page.locator("#previous-page").click();
      await page.waitForFunction(() => planInspector.pageIndex === 0);
    }
    const scopes = await page.evaluate(() =>
      [...planInspector.groups.values()].map((g) => ({
        id: g.id,
        pages: Math.ceil(g.children.length / planInspector.pageSize),
      })),
    );
    let maxVisible = 0,
      pagesChecked = 0;
    for (const { id, pages } of scopes)
      for (let index = 0; index < pages; index++) {
        await page.evaluate(
          ({ id, index }) => {
            location.hash =
              encodeURIComponent(id) + "&page=" + index + "&view=graph";
          },
          { id, index },
        );
        await page.waitForFunction(
          ({ id, index }) =>
            planInspector.scope === id &&
            planInspector.pageIndex === index &&
            planInspector.visibleCount > 0,
          { id, index },
        );
        const stats = await page.evaluate(() => {
          const data = JSON.parse(
              document.querySelector("#plan-data").textContent,
            ).graph,
            children = planInspector.visibleItems;
          const owner = new Map(
            children.flatMap((g) => g.leaves.map((i) => [i, g.id])),
          );
          const expected = new Set(
            children.flatMap((g) =>
              g.leaves.flatMap((i) =>
                data.nodes[i].inputs
                  .filter((j) => owner.has(j) && owner.get(j) !== g.id)
                  .map((j) => owner.get(j) + ">" + g.id),
              ),
            ),
          );
          const drawn = new Set(
            [...document.querySelectorAll(".edge")].map(
              (e) => e.dataset.from + ">" + e.dataset.to,
            ),
          );
          return {
            expected: [...expected].sort(),
            drawn: [...drawn].sort(),
            count: planInspector.visibleCount,
            children: children.length,
          };
        });
        assert.deepEqual(
          stats.drawn,
          stats.expected,
          "dependencies for " + id + " page " + index,
        );
        assert.equal(stats.count, stats.children);
        assert.ok(stats.count <= initial.pageSize);
        maxVisible = Math.max(maxVisible, stats.count);
        pagesChecked++;
      }
    // Search a unique full name, verify schema-derived bindings and output shape.
    const chosen = await page.evaluate(() => {
      const data = JSON.parse(document.querySelector("#plan-data").textContent);
      const i = data.node_metadata.findIndex((n) => n.bindings.length);
      return {
        i,
        name: data.graph.nodes[i].name,
        bindings: data.node_metadata[i].bindings,
        shape: planInspector.shape(i),
      };
    });
    await page.locator("#search").fill(chosen.name);
    await page.locator('.search-result[data-op="' + chosen.i + '"]').click();
    await page.waitForFunction(
      (i) => document.querySelector(".node.selected")?.dataset.id === "op:" + i,
      chosen.i,
    );
    const detail = await page.locator("#inspector").innerText();
    assert.ok(detail.includes(chosen.shape));
    for (const b of chosen.bindings) assert.ok(detail.includes(b.value));
    await page.locator("#search").fill("");
    if (screenshotDir)
      await page.screenshot({
        path: path.join(screenshotDir, "plan-operation.png"),
      });
    const refs = page.locator("#inspector [data-op]");
    if (await refs.count()) {
      const to = Number(await refs.first().getAttribute("data-op"));
      await refs.first().click();
      await page.waitForFunction(
        (i) =>
          document.querySelector(".node.selected")?.dataset.id === "op:" + i,
        to,
      );
      await page.goBack();
      await page.waitForFunction(
        (i) =>
          document.querySelector(".node.selected")?.dataset.id === "op:" + i,
        chosen.i,
      );
    }
    await page.locator("#search").fill("no-matching-operation-here");
    assert.equal(await page.locator(".search-result").count(), 0);
    await page.locator("#search").fill("");
    // Expand a module using the inspector, then return through the actual parent.
    const module = await page.evaluate(
      () => [...planInspector.groups.values()].find((g) => g.parent)?.id,
    );
    if (module) {
      await page.evaluate((id) => {
        const g = planInspector.groups.get(id);
        const index = g.parent.children.indexOf(g);
        location.hash =
          encodeURIComponent(g.parent.id) +
          "&page=" +
          Math.floor(index / planInspector.pageSize) +
          "&view=graph";
      }, module);
      await page.waitForFunction(
        (id) =>
          [...document.querySelectorAll(".node")].some(
            (n) => n.dataset.id === id,
          ),
        module,
      );
      await page
        .locator(".node")
        .filter({
          hasText: await page.evaluate(
            (id) => planInspector.groups.get(id).label,
            module,
          ),
        })
        .first()
        .click();
      await page.locator("#expand-module").click();
      await page.waitForFunction((id) => planInspector.scope === id, module);
      await page.locator("#up").click();
      await page.waitForFunction(() => planInspector.scope === "root");
    }

    await page.locator("#physical-mode").click();
    await page.waitForFunction(() => planInspector.physicalMode);
    const physicalInfo = await page.evaluate(() => {
      const p = JSON.parse(
        document.querySelector("#plan-data").textContent,
      ).physical;
      return { ranks: p.ranks.length, steps: p.schedules[0].steps.length };
    });
    assert.ok((await page.locator("[data-physical-step]").count()) <= 24);
    await page
      .locator("#physical-rank")
      .selectOption(String(physicalInfo.ranks - 1));
    await page.waitForFunction(
      (i) => planInspector.rankIndex === i,
      physicalInfo.ranks - 1,
    );
    if (physicalInfo.steps > 24) {
      await page.locator("#physical-next").click();
      await page.waitForFunction(() => planInspector.pageIndex === 1);
    }
    await page.locator("#physical-tab").selectOption("memory");
    assert.match(
      await page.locator("#workspace").innerText(),
      /Device HBM total: unresolved/,
    );
    await page.locator("[data-physical-step]").first().click();
    assert.match(
      await page.locator("#inspector").innerText(),
      /Physical decision/i,
    );
    assert.match(await page.locator("#inspector").innerText(), /Unknown/);
    assert.match(
      await page.locator("#inspector").innerText(),
      /Contract source fingerprint/i,
    );
    if (screenshotDir)
      await page.screenshot({
        path: path.join(screenshotDir, "plan-physical.png"),
      });
    await page.setViewportSize({ width: 760, height: 900 });
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
    );
    assert.deepEqual(errors, []);
    assert.deepEqual(requests, []);
    console.log(
      JSON.stringify({
        scopesChecked: scopes.length,
        pagesChecked,
        maxVisible,
        metadataHierarchy: "passed",
        navigation: "passed",
        bindingsAndShapes: "passed",
        browserErrors: 0,
        networkRequests: 0,
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
