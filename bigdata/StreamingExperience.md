#Hadoop Streaming 实际操作经验(主要是 python)


## 注意文件的可执行性
- 注意所有调用文件，在streaming调起时，用-file参数都标注上
- 注意所有调用文件，必须拥有可运行权限 chmod a+rx *.py
- python文件开头指定相关执行方式 “#!/usr/bin/env python”，或者直接stream里头直接写明“-mapper 'python wordcount_mapper.py'”

## 错误调试
- 展开Hadoop任务追踪的具体日志，一般可以看到具体哪一行python代码挂了，就知道自己的错误了
- 本地调试： 直接本地使用 “cat testfile* | ./mapper.py | sort | ./reducer.py” 可以小数据集上验证
- 注意使用常规的python指令，以前新的特性，很可能由于集群机器上的版本问题，无法执行（目测360的集群是2.6python）






