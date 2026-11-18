# DermAssist on a Hugging Face Docker Space (free CPU hardware).
# The Space only needs this Dockerfile and README.md: the code and the model are
# taken from the public GitHub repository when the Space is built.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user

RUN git clone --depth 1 https://github.com/subarna007/skin-lesion-classification-isic2018.git app

WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r app/requirements.txt

# GitHub limits files to 100 MB, so the checkpoint is stored in two parts: join and verify them now
RUN cat checkpoints/convnext_balanced.pth.part1 checkpoints/convnext_balanced.pth.part2 > checkpoints/convnext_balanced.pth \
    && echo "$(cat checkpoints/convnext_balanced.pth.sha256)  checkpoints/convnext_balanced.pth" | sha256sum -c - \
    && rm checkpoints/convnext_balanced.pth.part* \
    && mkdir -p app/data && chown -R user:user /home/user

USER user
ENV PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
