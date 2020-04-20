var_l=`env | awk -F "=" '{print $1}'`

for l in $var_l
do
    unset $l
done