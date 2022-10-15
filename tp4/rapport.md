# Compte rendu TP 4  - Représentation des connaissances

<p style="text-align:center">Corentin Giaufer--Saubert</p>


## Afficher toutes les femmes:

```sql
select ?women where {
    ?women prop:gender type:woman
}
```

| Num | ?women                                 |
| --- | -------------------------------------- |
| 1   | <http://uca.fr/family/member/Jessica>  |
| 2   | <http://uca.fr/staff/Alice>            |
| 3   | <http://uca.fr/family/member/Caroline> |
| 4   | <http://uca.fr/family/member/Victoria> |



## Afficher tous les professeurs:
    
```sql
select ?professor where {
    ?professor rdf:type job:teacher
}
```

| Num | ?professor                            |
| --- | ------------------------------------- |
| 1   | <http://uca.fr/family/member/Edouard> |
| 2   | <http://uca.fr/staff/Alice>           |
| 3   | <http://uca.fr/staff/Pierre>          |

## Afficher tous les professeurs et leur âge:
    
```sql
select ?professor ?age where {
    ?professor rdf:type job:teacher .
    ?professor prop:age ?age
}
```

| Num | ?professor                            | ?age |
| --- | ------------------------------------- | ---- |
| 1   | <http://uca.fr/family/member/Edouard> | 40   |


## Afficher tous les parents:
        
```sql
select distinct ?parent where {
    ?parent prop:child ?child

}
```

| Num | ?parent                                |
| --- | -------------------------------------- |
| 1   | <http://uca.fr/family/member/Edouard>  |
| 2   | <http://uca.fr/family/member/Jean>     |
| 3   | <http://uca.fr/family/member/Caroline> |


## Afficher tous les individus ayant une taille inférieure à 1.70m:
        
```sql
select ?individu where {
    ?individu prop:height ?height .
    filter(?height < 170)
}
```

| Num | ?individu                              |
| --- | -------------------------------------- |
| 1   | <http://uca.fr/family/member/Jean>     |
| 2   | <http://uca.fr/family/member/Caroline> |

## Afficher toutes les relations de parenté:
        
```sql
construct {
  ?x prop:parent ?name
} 
where {
  ?x prop:name ?name
}
```

| Num | ?x                                     | ?name    |
| --- | -------------------------------------- | -------- |
| 1   | <http://uca.fr/family/member/Edouard>  | Edouard  |
| 2   | <http://uca.fr/family/member/Jessica>  | Jessica  |
| 3   | <http://uca.fr/staff/Alice>            | Alice    |
| 4   | <http://uca.fr/staff/Pierre>           | Pierre   |
| 5   | <http://uca.fr/family/member/Jean>     | Jean     |
| 6   | <http://uca.fr/family/member/Caroline> | Caroline |
| 7   | <http://uca.fr/family/member/Yannick>  | Yannick  |
| 8   | <http://uca.fr/family/member/Victoria> | Victoria |

## Afficher les grands parents de Victoria:
        
```sql
select ?gp where {
    ?x prop:child family:Victoria .
    ?gp prop:child ?x 

}
```

| Num | ?gp                                    |
| --- | -------------------------------------- |
| 1   | <http://uca.fr/family/member/Jean>     |
| 2   | <http://uca.fr/family/member/Caroline> |

## Demander si Jean est le grand-père de Victoria:
        
```sql
ask {
    ?x prop:child family:Victoria .
    family:Jean prop:child ?x 
}
```

```xml
<?xml version="1.0" ?>
<sparql xmlns='http://www.w3.org/2005/sparql-results#'>
<head>
</head>
<boolean>true</boolean>
</sparql>
```