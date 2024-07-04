### 题目链接
https://leetcode.cn/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/

### 解题思路

创建一个哈希表来记录数组中每个元素的出现次数，返回第一个出现次数大于2的数。



### 相关代码

```java
class Solution {
    public int findRepeatDocument(int[] documents) {
        Map<Integer,Integer> hash = new HashMap<>();
        for(int i=0;i<documents.length;i++){
            if(hash.get(documents[i])==null){
                hash.put(documents[i],1);
            }
            else{
                return documents[i];
            }
        }
        return documents[0];
    }
}
```

