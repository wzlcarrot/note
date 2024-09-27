## 简介

ssh是用来连接服务器的一种工具。



## 配置别名

1. 在本地配置（windows版）

```
路径:C:\Users\10179\.ssh
```

```
编写 config
```

config文件

```
Host my_server
	HostName ip地址
	User wzlcarrot
	Port 22
```

2. 在本地配置（linux版）

```
路径:在用户目录下，比如在wzlcarrot目录下
mkdir .ssh
```

```
vim config
```

config文件

```
Host my_server
	HostName ip地址
	User wzlcarrot
	Port 22
```

**配置完成后，下一次可以使用ssh my_server来连接**



## ssh免密

1. 在.ssh目录下执行以下命令（windows版）

```
路径:C:\Users\10179\.ssh
ssh-keygen -t rsa
```

在linux下也是同样操作。

2. 在服务器上创建.ssh目录。

```
mkdir .ssh
```

3. 创建authorized_keys文件

4. 把本地的id_rsa.pub文件内容复制到authorized_keys文件中
5. 重新登录，已成功。



