
# Dependencies

To obtain the results described in the paper, you should first install the python dependencies:

```python
pip3 install -r requirements.txt
```

Moses and MGIZA also need to be installed (follow this tutorial: http://www.statmt.org/moses/?n=Development.GetStarted)

Finally, update the paths in `paths.py` and run the `main.sh` scripts

# Execution

To run the models described on the paper to obtain the results for the Semeval datasets, run the following script:

```bash
./main.sh
```

# Evaluation

The results obtained by the command described in the previous section are available on the `results/` folder.
To obtain the evaluation metrics depicted in the paper based on these results, run the following script:

```python
python3 results.py
```

**Author:** Thiago Castro Ferreira