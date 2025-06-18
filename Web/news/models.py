from django.db import models

class News(models.Model):
    title = models.CharField(max_length=200, verbose_name='标题')
    content = models.TextField(verbose_name='内容')
    pub_date = models.DateTimeField(auto_now_add=True, verbose_name='发布时间')
    model_type = models.CharField(max_length=10, choices=[('SVM', 'SVM模型'), ('LSTM', 'LSTM模型')], verbose_name='模型类型')

    class Meta:
        verbose_name = '新闻'
        verbose_name_plural = verbose_name
        ordering = ['-pub_date']

    def __str__(self):
        return self.title