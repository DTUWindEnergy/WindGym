/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    // Getting Started
    "index",
    "installation",

    // Core Documentation
    {
      type: "category",
      label: "User Guide",
      collapsible: true,
      collapsed: false,
      items: [
        "concepts",
        "simulations",
        "agents",
        "evaluations",
        "noise-and-uncertainty",
      ],
    },

    // Reference
    {
      type: "category",
      label: "Reference",
      collapsible: true,
      collapsed: false,
      items: ["api-reference", "troubleshooting"],
    },

    // Contributing
    "developer-guidelines",
  ],
};

module.exports = sidebars;
