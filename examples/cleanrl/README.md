# Examples

To run CleanRL example, run

```bash
pip install -r requirements.txt
```

then: 

```bash
python ppo.py
```

or, to run on a Slurm system quickly:

```bash
sbatch --time=1-0 -p gaia --exclusive --wrap="conda run -n cee python ppo.py" 
```

replacing `cee` with the name of your Anaconda environment.