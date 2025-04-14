# openbayes 自动调参样例

推荐使用 PyTorch 版本的示例代码（位于 `/pytorch` 目录下）。TensorFlow 版本已不推荐使用。

```text
# 1. init gear
bayes gear init <container-id>

# 2. create hypertuning
bayes gear run hypertuning

# 3. show result
bayes container open <container-id>
```

详细的信息参考 [openbayes 自动调参](https://openbayes.com/docs/hypertuning/)
