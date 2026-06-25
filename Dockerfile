# Credit Risk Studio — AWS Lambda container image (Phase 3)
FROM public.ecr.aws/lambda/python:3.12

# Install runtime dependencies into the Lambda task root.
COPY requirements-lambda.txt ./
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Application code.
COPY creditrisk/ ${LAMBDA_TASK_ROOT}/creditrisk/
COPY app/        ${LAMBDA_TASK_ROOT}/app/

# matplotlib/seaborn are imported at module load; on Lambda only /tmp is
# writable, so point the config dir there and use the headless backend.
ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

# Mangum ASGI adapter exposed as app.main.handler.
CMD ["app.main.handler"]
