// Generated using webpack-cli https://github.com/webpack/webpack-cli

const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const { DefinePlugin } = require("webpack");

const isProduction = process.env.NODE_ENV == "production";

const stylesHandler = "style-loader";

const DEVSERVER_PORT=9092
const BACKEND_SOCKET_PORT=8765
const BACKEND_SOCKET_URL= process.env.CODESPACES ? `wss://${process.env.CODESPACE_NAME}-${BACKEND_SOCKET_PORT}.${process.env.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}` : `ws://localhost:${BACKEND_SOCKET_PORT}`;

const config = {
  entry: "./src/index.js",
  output: {
    path: path.resolve(__dirname, "dist"),
  },
  devServer: {
    client: {
      webSocketURL: {
        port: process.env.CODESPACES ? 443 : DEVSERVER_PORT,
      }
    },
    open: false,
    host: '0.0.0.0',
    port: DEVSERVER_PORT,
    allowedHosts: ['localhost', `.${process.env.GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}`],
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "index.html",
    }),

    // Add your plugins here
    // Learn more about plugins from https://webpack.js.org/configuration/plugins/
    new DefinePlugin({
      "webpack_env.BACKEND_SOCKET_URL": JSON.stringify(BACKEND_SOCKET_URL),
    }),
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/i,
        loader: "babel-loader",
      },
      {
        test: /\.css$/i,
        use: [stylesHandler, "css-loader"],
      },
      {
        test: /\.(eot|svg|ttf|woff|woff2|png|jpg|gif)$/i,
        type: "asset",
      },

      // Add your rules for custom modules here
      // Learn more about loaders from https://webpack.js.org/loaders/
    ],
  },
};

module.exports = () => {
  if (isProduction) {
    config.mode = "production";
  } else {
    config.mode = "development";
  }
  return config;
};
