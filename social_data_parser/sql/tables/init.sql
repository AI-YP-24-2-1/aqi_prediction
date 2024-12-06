create table if not exists peverty_level(
    region text not null,
    year date not null,
    value real not null
)
-- value - значение в процентах от общей численности населения

alter table peverty_level add primary key (region, year)

