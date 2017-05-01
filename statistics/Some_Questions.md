#Game Show about Bayes Theorem
如果A事件发生的概率是1/3
如果B事件发生的概率是1/3
如果C事件发生的概率是1/3
如果D事件发生的概率是1/3

那么，同时发生三件事的概率是多少？

解答：
假设独立，那概率就是C(3,4)*(1/3)^3*(2/3)=8/81


（请注意排列组合的基本公式...）




那我等答案间隙再出个和编程相关的概率题：假设有个可以等概率生成0-4的函数Random5()，问我们应该如何调用它来写个Random7()和Random1000()，既一个能等概率生成“0～数字减1”的新函数

解答：
设已有随机数产生器 $a$ 均匀地生成 $[0, a-1]$；与推广为产生器 $b (b>a)$。
则级联 $a$ 可以找到 $\min_{n}{a^n>b}$；设 $l=floor(a^n/b)$，当 $a^n$ 产生的随机数大于 $bl$ 时需要重新产生，否则直接输出 $a^n\ mod\ b$ 即可；一次性产出结果概率为 $\frac{bl}{a^n}$



select * from scores where arbiter_id not in (
    select
      arbiter_id
    from (
        select
          arbiter_id,
          count(*) as nu
        from (
            select arbiter_id from scores where first_criterion in (
              select max(first_criterion) from scores
            ) union all
            select arbiter_id from scores where first_criterion in (
              select min(first_criterion) from scores
            ) union all
            select arbiter_id from scores where second_criterion in (
              select max(second_criterion) from scores
            ) union all
            select arbiter_id from scores where second_criterion in (
              select min(second_criterion) from scores
            ) union all
            select arbiter_id from scores where third_criterion in (
              select max(third_criterion) from scores
            ) union all
            select arbiter_id from scores where third_criterion in (
              select min(third_criterion) from scores
            )
        ) A
        group by arbiter_id
    ) B
    where nu>=2
);




