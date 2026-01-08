from itertools import chain, combinations, combinations_with_replacement
import itertools
results = {0:'ARGENTINA_WIN',1 :'DRAW&ARG',2:'DRAW&FRA',3:'FRANCE_WIN'}
people = {0:'MAOR',1:'YARDEN',2:'DOR'}
possible = [0,1,2,3]
points ={0:66,1:74,2:64}
outcomes = ['ARG','DRAW','FRA']
comb = [p for p in itertools.product(outcomes, repeat=3)]
print(comb)
result_to = {'ARG':[0],'DRAW':[1,2],'FRA':[3]}
avg_points = {0:[0,0,0],1:[0,0,0],2:[0,0,0]}
maor_win_cases = 0
for result in possible:
    for c in comb:
        print(result,c)
        for i,r in enumerate(c):
            if result in result_to[r]:
                avg_points[outcomes.index(c[0])][i]+=(8-4*int(result in [1,2]))/(len(comb))
    if(result in [0,1]):
        avg_points[0][0]+=5/4
        avg_points[1][0]+=2.5/4
        avg_points[2][0]+=5/4
    if(result in [2,3]):
        avg_points[0][2]+=5/4
        avg_points[1][2]+=2.5/4
        avg_points[2][2]+=5/4
print(avg_points)


# for c in combinations_with_replacement(possible,3):
#     for results in possible:
