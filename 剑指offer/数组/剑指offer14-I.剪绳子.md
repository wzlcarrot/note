## 题目链接

https://leetcode.cn/problems/jian-sheng-zi-lcof/submissions/544804266/



## 解题思路

完全背包问题模型。



## 相关代码

```vue
class Solution {
    public int cuttingBamboo(int bamboo_len) {
        //f[i][j] i表示的是物体的种类，j表示的是物体的体积
        int f[][] = new int[60][60];
        for(int i=0;i<60;i++){
            for(int j=0;j<60;j++){
                f[i][j] = 1;
            }
        }
        for(int i=1;i<bamboo_len;i++){
            for(int j=0;j<=bamboo_len;j++){
                for(int k=0;j>=k*i;k++){
                    f[i][j] = Math.max(f[i][j],f[i-1][j-k*i]*(int)Math.pow(i,k));
                }
            }
        }      

        return f[bamboo_len-1][bamboo_len];
    }
}
```

