// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const { themes } = require("prism-react-renderer");
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.nightOwl;
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "WindGym Docs",
  tagline: "documentation of WindGym package",
  url: "https://pages.windenergy.dtu.dk",
  baseUrl: "/windgym/",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/favicon.ico",
  // Use plain markdown format to avoid MDX parsing issues with auto-generated API docs
  markdown: {
    format: "md",
  },
  // GitHub pages deployment config.
  organizationName: "DTU Wind Energy Systems",
  projectName: "WindGym",
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },
  presets: [
    [
      "@docusaurus/preset-classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: "/", // Serve docs from the root
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl:
            "https://gitlab.windenergy.dtu.dk/sys/windgym/-/edit/main/Docs",
          // === TEMPORARILY REMOVE ALL remarkPlugins and rehypePlugins FOR TESTING ===
          // remarkPlugins: [], // No plugins for now
          // rehypePlugins: [], // No plugins for now
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],
  // === TEMPORARILY REMOVE MATH STYLESHEET FOR TESTING ===
  stylesheets: [
    // {
    //   href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
    //   type: 'text/css',
    //   integrity: 'sha384-Tw7uHKUwpYad+s7fdlA0PlsHQJpgbFVgYwNwuS/nD6WbOWbKijxX+v+rUtyzT6b0',
    //   crossorigin: 'anonymous',
    // },
  ],
  // === END MATH SUPPORT ADDITION ===
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "WindGym Docs",
        items: [
          {
            type: "doc",
            docId: "index",
            position: "left",
            label: "Documentation",
          },
          {
            type: "doc",
            docId: "api/index",
            position: "left",
            label: "API Reference",
          },
          {
            href: "https://gitlab.windenergy.dtu.dk/sys/windgym/-/tree/main/examples",
            label: "Examples",
            position: "left",
          },
          {
            href: "https://gitlab.windenergy.dtu.dk/sys/windgym",
            label: "GitLab",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              {
                label: "Getting Started",
                to: "/installation",
              },
              {
                label: "User Guide",
                to: "/concepts",
              },
              {
                label: "API Reference",
                to: "/api/",
              },
              {
                label: "Troubleshooting",
                to: "/troubleshooting",
              },
            ],
          },
          {
            title: "Community",
            items: [
              {
                label: "GitLab",
                href: "https://gitlab.windenergy.dtu.dk/sys/windgym",
              },
              {
                label: "Issues",
                href: "https://gitlab.windenergy.dtu.dk/sys/windgym/-/issues",
              },
              {
                label: "Discussions",
                href: "https://gitlab.windenergy.dtu.dk/sys/windgym/-/discussions",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "Examples",
                href: "https://gitlab.windenergy.dtu.dk/sys/windgym/-/tree/main/examples",
              },
              {
                label: "DTU Wind Energy",
                href: "https://windenergy.dtu.dk/",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Technical University of Denmark (DTU). Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};
module.exports = config;
