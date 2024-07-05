### 题目链接

https://leetcode.cn/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/submissions/544373791/



### 解题思路

遍历一下数组，得到最小元素。



### 相关代码

```java
class Solution {
    public int stockManagement(int[] stock) {
        int res = Integer.MAX_VALUE;
        for (int j : stock) {
            res = Math.min(res, j);
        }
        return res;
    }

}
```

