# multi-fidelity symbolic regression

A repo to start the Symbolic regression for multi-fidelity project. This is still work in-progress in my leisure time, so no any guarantee.

Symbolic regression relies on MilesCranmer PySR: https://github.com/MilesCranmer/PySR

Suggestion for low-fidelity fitting using model_selection = "best",
high-fidelity fitting using model_selection = "score" (avoiding relying too much on minimizing the loss).

## Model: As simple as it is

Basically, fit the LF, predict from LF, propagate to

```python

class MultiFidelitySRegressor:
    """
    Multi-fidelity surrogate model that predicts the high-fidelity function.

    Parameters
    ----------
    model : PySRRegressor
        Low-fidelity model.
    model_mf : PySRRegressor
        Multi-fidelity model.
    complexity : int
        Complexity of the multi-fidelity model.
    """
    def __init__(self, model, model_mf, complexity=1):
        self.model = model
        self.model_mf = model_mf
        self.complexity = complexity

    def fit(self, x_l, y_l, x_h, y_h):
        """
        Fit the multi-fidelity model.

        Parameters
        ----------
        x_l : array-like
            Low-fidelity input data.
        y_l : array-like
            Low-fidelity target data.
        x_h : array-like
            High-fidelity input data.
        y_h : array-like
            High-fidelity target data.
        """
        # train LF model
        self.model.fit(x_l, y_l)
        # Predict LF model on HF input data
        f_l = self.model.predict(x_h)

        # Stack the LF model predictions with the HF input data
        _x_h = np.hstack([x_h, f_l[:, None]])

        self.model_mf.fit(_x_h, y_h)

    def predict(self, x):
        """
        Predict the high-fidelity function.
        """
        f_l = self.model.predict(x)
        _x = np.hstack([x, f_l[:, None]])
        return self.model_mf.predict(_x, self.complexity)

    def sympy(self):
        return self.model.sympy(), self.model_mf.sympy(self.complexity)

    def __call__(self, x):
        return self.predict(x)

    def __repr__(self):
        return f"MultiFidelitySRegressor(model={self.model}, model_mf={self.model_mf}, complexity={self.complexity})"
```


## Example taken from [Adapt website](https://adapt-python.github.io/adapt/examples/Multi_fidelity.html)
It somehow works for missing data in high-fidelity, but the output is sensitive to the model selection method you use. I use model="score" here, which works better than model="best".


![Image](https://github.com/user-attachments/assets/d5f13296-569c-43f4-9fc5-346a58e8a57d)

## Example taken from Emukit website (Forrester function)

This is the most difficult example, but also shows the limitation of SR on this type of problem. I guess the range of x is too short for SR to understand which function it is.

![Image](https://github.com/user-attachments/assets/1a8ce773-4239-4a01-83e4-89cfae73ae46)

## Another example taken from Emukit website

It works perfectly well for sin/cos function.

![Image](https://github.com/user-attachments/assets/ff6a50de-e0f5-4c89-8665-c3a5a1bba794)