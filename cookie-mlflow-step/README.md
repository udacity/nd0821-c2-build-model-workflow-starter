# Cookiecutter template for MLFlow steps

Using this template you can quickly generate new steps to be used with MLFlow.

# Usage
Run the command:

```
> cookiecutter [path to this repo] -o [destination directory]
```

and follow the prompt. The tool will ask for the step name, the script name, the description and so on. It will
also ask for the parameters of the script. This should be a comma-separated list of parameter names *without spaces*. 
After you are done, you will find a new directory with the provided step name containing a stub of a new MLflow step.

You will need to edit both the script and the MLproject files to fill in the type and the help for the parameters.
Of course, if your script needs packages, these should be added to the conda.yml file.