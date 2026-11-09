FROM rocker/r-ver:4.6.1

WORKDIR /app

COPY src/R_Scripts/install_packages.R .
RUN Rscript install_packages.R

COPY src/ ./src/

CMD ["Rscript", "src/R_Scripts/main.R"]