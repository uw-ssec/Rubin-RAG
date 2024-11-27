ARG PIXI_TAG=0.38.0-noble
FROM ghcr.io/prefix-dev/pixi:$PIXI_TAG

# Specify arg again here
ARG PIXI_TAG

# run the `install` command (or any other). This will also install the dependencies into `/app/.pixi`
# assumes that you have a `prod` environment defined in your pixi.toml
RUN pixi install
# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e prod > /shell-hook.sh

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh
