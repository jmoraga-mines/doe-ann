#!/bin/bash
srun -p sgpu-testing --pty --mem 32G -t 30:00 /bin/bash
