FROM python:latest
RUN pip install matplotlib
WORKDIR /app
COPY verify_test_bib_numbers.py ./
COPY scratch.py ./
# COPY matplotlibrc /root/.config/matplotlib/matplotlibrc
# ENV MATPLOTLIBRC=/root/.config/matplotlib
# CMD ["python", "./verify_test_bib_numbers.py", "./images/"]
CMD ["python", "./scratch.py"]
