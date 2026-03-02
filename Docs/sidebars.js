/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    // Getting Started
    "index",
    "installation",
    "quick-start",

    // Core Documentation
    {
      type: "category",
      label: "User Guide",
      collapsible: true,
      collapsed: false,
      items: [
        "concepts",
        "rewards",
        "simulations",
        "agents",
        "evaluations",
        "noise-and-uncertainty",
      ],
    },

    // API Reference
    {
      type: "category",
      label: "API Reference",
      collapsible: true,
      collapsed: false,
      link: {
        type: "doc",
        id: "api/index",
      },
      items: [
        "api/modules/windgym",
        "api/modules/core",
        "api/modules/agents",
        "api/modules/wrappers",
        "api/modules/visualization",
        "api/modules/utils",
      ],
    },

    // Other Reference
    "troubleshooting",

    // Contributing
    "developer-guidelines",
  ],
};

module.exports = sidebars;
