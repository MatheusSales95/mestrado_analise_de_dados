# Roteiro: área favorável à erva-mate com Random Forest e validação espacial

Atividade prática de CAP-423. Alvo: erva-mate (*Ilex paraguariensis*), já registrada na planilha (Erva-mate, GBIF = SIM).

**Prazo:** 18/11/2026, às 23h59. O seminário 3 (João e Matheus) é em 16/11, com slides até 15/11, então a meta é terminar até 14/11.

**Sobre os números:** os blocos de código deste roteiro foram executados em sequência, junto com as seções indicadas do notebook da aula, em 30/10/2026, no ambiente `base`. Os valores marcados como "piloto" servem para conferir se você está no caminho. O formulário deve receber os valores da sua execução final, porque o GBIF muda com o tempo.

## Visão geral

O enunciado cita o notebook `05_Araucaria_RF_Classification_SpatialCV`. No material, ele é o `06/08_Araucaria_RF_Classification_SpatialCV.ipynb`. Os caminhos abaixo são relativos a `materialApoio/`. O notebook não lê nada de lá: tudo o que ele usa fica em `atvErvaMate/input/`.

| Etapa | Item do enunciado | Base no material | O que muda em relação à aula |
|---|---|---|---|
| 0. Ambiente e dados | – | `README.md` | Já resolvida: `scikit-learn` instalado e contorno do IBGE no lugar do `brazil.gpkg` |
| 1. Download | 2 | Enunciado | Buscar por `scientificName` e paginar |
| 2. Preditores | 5 | `05/feature-engineering/knn-interpolation.ipynb` | Lista fixa de rasters e solo por KNN |
| 3. Limpeza | 3 | `06/06`, seção 5 | Acrescentar (0,0), Brasil, pixel e erros encontrados nos dados |
| 4. Pseudoausências | 4 | `06/06`, seções 6 a 12 | Sortear pixels válidos, não pontos no polígono |
| 5. Modelagem | 5 | `06/08`, seções 1 a 14 | **Substituir a alocação de folds, que falha com 10 folds** |
| 6. Mapas | 5 | `06/08`, seções 15 a 19 | Trocar nomes de arquivos e títulos |
| 7. Entrega | 6 | – | Resumo para o formulário e `.zip` |

O notebook `atvErvaMate/ervaMate.ipynb` já está completo: as Etapas 0 a 7 rodaram do início ao fim em um kernel `base` novo, em 30/10/2026. Os blocos abaixo são as células principais, na ordem em que aparecem.

## Etapa 0: ambiente e dados

Tudo roda localmente, no kernel `base` (Miniforge, Python 3.13). Nada depende do BDC-Lab, e esta etapa já está pronta:

| Item | Situação |
|---|---|
| Dependências | `scikit-learn` 1.9.1 instalado no `base`, que já tinha `pygbif`, `rasterio` e `geopandas` |
| Rasters e pontos de solo | Em `atvErvaMate/input/`, copiados de `materialApoio/05/input/` |
| Contorno do Brasil | Em `atvErvaMate/input/brazil.gpkg`, gerado da malha oficial do IBGE |

**Declare a troca do contorno no notebook.** O enunciado pede o `brazil.gpkg` do curso, que só existe no BDC-Lab. O arquivo local tem o mesmo nome, mas vem da [malha `BR_Pais_2025` do IBGE](https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2025/Brasil/BR_Pais_2025.zip), convertida de SIRGAS 2000 para EPSG:4326. Se você conseguir o arquivo do curso, basta sobrescrevê-lo: o código não muda, mas as contagens e o AUC podem mudar, porque o sorteio das pseudoausências depende do contorno.

**Conferência do contorno (30/10/2026):**

- As distâncias entre os extremos batem com o [mapa do IBGE](https://geoftp.ibge.gov.br/produtos_educacionais/atlas_educacionais/atlas_geografico_escolar/mapas_do_brasil/mapas_nacionais/fisico/brasil_pontos_extremos_e_fronteiras.pdf): 4.378,1 km norte-sul (IBGE: 4.378,4) e 4.326,1 km leste-oeste (IBGE: 4.326,6).
- 22 dos 25 preditores têm a mesma máscara, e 99,84 % dos pixels dela ficam dentro do contorno. Na área natural da erva-mate, os pixels que ficam fora estão a no máximo 2,2 pixels dele.
- Dentro do contorno, os pixels sem dado são sobretudo lagoas e baías que a malha do IBGE inclui e os rasters não cobrem, como a Lagoa dos Patos. A máscara de pixels válidos já os exclui.
- A malha traz Fernando de Noronha, Trindade e Martim Vaz, mas não o Atol das Rocas nem São Pedro e São Paulo. Isso não afeta o trabalho, porque essas ilhas ficam fora da grade dos rasters.

Mantenha esta primeira célula: no seu ambiente ela não altera nada, e em outro ela instala o que falta. Para instalar outros pacotes, use `%pip`, que instala no Python do kernel. O `pip` do terminal pertence a outro Python (3.12), e `conda install` no `base` pode sobrescrever o numpy e o matplotlib, que vieram do pip.

```python
%pip install -q pygbif scikit-learn
```

Depois desta célula, cole a célula de imports do `06/08` e, em seguida, a configuração abaixo. O desvio para a pasta do BDC-Lab só serve para o caso de o professor abrir o notebook lá. Você não precisa rodar nada lá, e esse caminho não foi testado.

```python
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from pygbif import occurrences as occ
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from sklearn.neighbors import KNeighborsRegressor

# Os arquivos locais ficam todos na pasta do notebook. O desvio só vale se ele for aberto no BDC-Lab.
BDC_INPUT = Path("/home/jovyan/grupos/pgrad-disciplinas/CAP-423-ciencia-de-dados-geoespaciais/dataset/input")
INPUT_DIR = BDC_INPUT if BDC_INPUT.exists() else Path("input")
BRAZIL_PATH = INPUT_DIR / "brazil.gpkg"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 123
BUFFER_KM = 20
METRIC_CRS = "EPSG:5880"  # mesmo CRS métrico do notebook 06/06
```

```python
# Conferência visual do contorno usado na limpeza e no sorteio das pseudoausências.
ax = gpd.read_file(BRAZIL_PATH).plot(figsize=(7, 7), facecolor="#cde2fb", edgecolor="#2a78d6", linewidth=0.5)
ax.set(title="Contorno do Brasil", xlabel="Longitude", ylabel="Latitude")
plt.show()
```

## Etapa 1: baixar as ocorrências

Use `scientificName`, como no enunciado, em vez do `taxonKey=5414252`. O `taxonKey` traz 1608 registros e o nome traz 1599, incluindo os sinônimos. Os 9 a mais têm o código `SH1046288.09FU` e vêm de amostras de solo (Tedersoo et al.), então não são observações da planta. O nome também elimina o `DeprecationWarning`.

```python
arquivo_bruto = OUTPUT_DIR / "gbif_bruto.csv"

if arquivo_bruto.exists():
    raw = pd.read_csv(arquivo_bruto, low_memory=False)
else:
    registros, offset = [], 0
    while True:
        r = occ.search(
            scientificName="Ilex paraguariensis", country="BR",
            hasCoordinate=True, hasGeospatialIssue=False,
            limit=300, offset=offset,
        )
        registros.extend(r["results"])
        offset += 300
        if r["endOfRecords"]:
            break
    raw = pd.DataFrame(registros)
    raw.to_csv(arquivo_bruto, index=False)

# A ordem fixa torna reprodutível a escolha do registro mantido em cada pixel.
raw = raw.sort_values("key").reset_index(drop=True)
print("Registros baixados:", len(raw))
```

**Piloto:** 1599 registros. O CSV evita novos downloads enquanto você desenvolve. Busca com `occ.search` não gera DOI, e o enunciado só pede o DOI "quando houver". Fique com ela, porque o notebook roda sem credenciais do GBIF.

## Etapa 2: preditores e máscara de pixels válidos

A lista fixa evita capturar outros `.tif` da pasta. Ela também mantém a ordem das colunas entre Windows e Linux, o que o `glob` da aula não garante: o `sorted` coloca `HAND` primeiro no Linux e `bio1` no Windows.

```python
PREDITORES = [f"bio{i}" for i in range(1, 20)] + [
    "elevation", "slope", "HAND", "TPI_11x11", "NTSI", "Mean_curvature",
]
raster_files = [INPUT_DIR / f"{nome}.tif" for nome in PREDITORES]

# Pixel válido = dado presente em todos os preditores.
valid = None
for path in raster_files:
    with rasterio.open(path) as src:
        arr = src.read(1)
        ok = np.isfinite(arr)
        if src.nodata is not None:
            ok &= arr != src.nodata
        valid = ok if valid is None else valid & ok

with rasterio.open(raster_files[0]) as ref:
    transform, perfil = ref.transform, ref.profile.copy()

print(f"Pixels válidos: {valid.sum():,} de {valid.size:,}")
```

**Piloto:** 10.120.871 de 22.024.128 pixels (grade de 4704 × 4682, cerca de 1 km).

**Faixa em branco nos mapas.** `HAND`, `NTSI` e `slope` não têm dado em parte do litoral e da foz do Amazonas. Isso tira da máscara 94.103 pixels dentro do contorno que os outros 22 preditores cobrem. Essas áreas ficam sem amostras e em branco nos mapas, e o motivo não é o contorno.

### Solo

O enunciado cita "dados de solo", mas os 25 rasters da aula não incluem solo, e o material local só o tem como pontos (`c3_soildata_psd_modeling_*.gpkg`). Por isso, gere os rasters com o KNN da aula (`knn-interpolation.ipynb`) e diga no notebook que eles vêm desses pontos. Como a pasta do BDC-Lab não será consultada, não dá para saber se ela tem rasters de solo prontos.

Há duas diferenças em relação à aula. A grade é a dos preditores, não uma de 10 km. E não há máscara de 100 km, porque só 3,8 % dos pixels válidos estão além dessa distância (máximo de 199 km) e a máscara abriria buracos no mapa.

```python
soil = gpd.read_file(INPUT_DIR / "c3_soildata_psd_modeling_meters_0_50cm.gpkg")
soil = soil[soil["esqueleto"] < 1000]  # mesma regra de 05/01_EDA
FRACOES = ["areia", "silte", "argila"]

knn = KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
knn.fit(np.column_stack([soil.geometry.x, soil.geometry.y]), soil[FRACOES].to_numpy())

linhas_v, colunas_v = np.nonzero(valid)
lons, lats = xy(transform, linhas_v, colunas_v)
xm, ym = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True).transform(
    np.asarray(lons), np.asarray(lats)
)
pixels = np.column_stack([xm, ym])
pred = np.vstack([
    knn.predict(pixels[i:i + 1_000_000]) for i in range(0, len(pixels), 1_000_000)
])

perfil.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
for j, nome in enumerate(FRACOES):
    grade = np.full(valid.shape, np.nan, dtype="float32")
    grade[linhas_v, colunas_v] = pred[:, j]
    with rasterio.open(OUTPUT_DIR / f"{nome}.tif", "w", **perfil) as dst:
        dst.write(grade, 1)

# A lista é refeita, e não acrescida, para a célula poder rodar de novo sem duplicar o solo.
raster_files = [INPUT_DIR / f"{nome}.tif" for nome in PREDITORES]
raster_files += [OUTPUT_DIR / f"{nome}.tif" for nome in FRACOES]
print("Preditores:", len(raster_files))
```

**Piloto:** cerca de 20 s. Os três rasters têm dado em todos os pixels válidos, então a máscara não muda.

**Decisão: manter o solo no modelo.** O enunciado o inclui, e o custo é pequeno. No piloto, o efeito foi pequeno: AUC do teste de 0,975 sem solo e 0,982 com solo, na mesma divisão, uma diferença menor que a variação entre sementes. A média na validação espacial foi a mesma (0,974), e a variável de solo mais bem colocada, `areia`, ficou em 11º lugar entre 28. Relate isso com duas ressalvas: KNN sobre pontos esparsos é uma camada grosseira, e aqui só há textura, enquanto o [estudo da Embrapa](https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/300244/1/AreaDistribuicao.pdf) associa a espécie a baixo teor de nutrientes e alto alumínio. O resultado, portanto, não prova que o solo seja irrelevante. Se preferir remover, essa é a justificativa.

## Etapa 3: limpeza

A limpeza da aula (`06/06`, seção 5) só remove geometria vazia e coordenada repetida. O enunciado exige mais, e os dados da erva-mate têm erros que justificam regras adicionais.

| Regra adicional | Evidência nos dados | Registros | Pixels |
|---|---|---|---|
| Centro do Brasil | 75 exemplares de 1816 a 1998, sem estado nem município, todos em (−10,8339; −52,8731), e mais 3 em (−10,7731; −53,0898). Os dois pontos estão a menos de 60 km do centro geométrico do país, o que indica georreferência no nível do país | 78 | 2 |
| Incerteza > 10 km | Inútil com pixel de 1 km. Inclui os 53 registros do iNaturalist, com cerca de 30 km | 85 | 76 |
| Longitude trocada | Rio de Contas (BA) com longitude −47,95. Os demais registros do município estão em cerca de −41,9 | 2 | 2 |
| Fora da distribuição | Amazonas (1971) e Sinop/MT (2017), regiões que o estudo da Embrapa não cita | 4 | 3 |
| Grau inteiro | (−14,0003; −50,0004), em Goiás (1966) | 1 | 1 |

Três escolhas para deixar claras no notebook:

- **Incerteza ausente não exclui.** Ela falta em cerca de 94 % dos registros, então exigi-la eliminaria quase toda a base.
- **O Distrito Federal permanece** (5 registros, que viram 3 presenças). O estudo da Embrapa diz que os limites da espécie "são ainda indeterminados", então não há evidência suficiente para excluí-lo.
- **Não sobra presença em Goiás.** A única coordenada no estado é a do ponto de grau inteiro. Outro registro marcado como Rio Verde/GO (1997) tem coordenada no sul de São Paulo (−24,08; −49,21). Ele permanece, porque não dá para saber qual campo está errado e a coordenada cai na área principal da espécie.
- **A lista oficial de estados continua sem confirmação.** Não consegui acessar a [Flora e Funga do Brasil](https://floradobrasil.jbrj.gov.br). Um resumo de busca de uma [página da UFMG](https://www.ufmg.br/mhnjb/ceplamt/?p=1354), que cita a Flora do Brasil 2020 e que também não abriu, lista BA, DF, MS, MT, MG, SP, PR, RS e SC. Confira a lista oficial. Se ela confirmar Mato Grosso, decida se os 2 registros de Sinop (a 11,6°S) voltam: a lista por estado, sozinha, não valida esse ponto.
- **A *var. sincorensis* permanece** (83 registros, quase todos na Bahia), porque o alvo é a espécie e o GBIF a trata assim. O estudo da Embrapa não cita a Bahia, então essa escolha modela a espécie, não apenas a área ervateira clássica. Ela causa a mancha da Chapada Diamantina no mapa, que não é erro.

```python
brazil = gpd.read_file(BRAZIL_PATH).to_crs("EPSG:4326")
brasil = brazil.geometry.union_all()
shapely.prepare(brasil)  # acelera as consultas ponto no polígono


def dentro_do_brasil(g):
    return shapely.contains_xy(brasil, g.geometry.x.to_numpy(), g.geometry.y.to_numpy())


etapas = {"Baixados do GBIF": len(raw)}

pres = raw.dropna(subset=["decimalLatitude", "decimalLongitude"])
pres = pres[~((pres["decimalLatitude"] == 0) & (pres["decimalLongitude"] == 0))]
pres = gpd.GeoDataFrame(
    pres,
    geometry=gpd.points_from_xy(pres["decimalLongitude"], pres["decimalLatitude"]),
    crs="EPSG:4326",
)
etapas["Com coordenada válida, sem (0,0)"] = len(pres)

lat, lon = pres.geometry.y.round(3), pres.geometry.x.round(3)
incerteza = pd.to_numeric(pres["coordinateUncertaintyInMeters"], errors="coerce")
regras = {
    "Centro do Brasil": ((lat == -10.834) & (lon == -52.873)) | ((lat == -10.773) & (lon == -53.090)),
    "Incerteza > 10 km": incerteza > 10_000,
    "Longitude trocada": pres["municipality"].fillna("").str.contains("Rio de Contas") & (lon < -44),
    "Fora da distribuição": (lon < -60) | (lat.between(-12, -11) & lon.between(-56, -55)),
    "Grau inteiro": (lat == -14.0) & (lon == -50.0),
}
print({nome: int(m.sum()) for nome, m in regras.items()})
pres = pres[~np.logical_or.reduce(list(regras.values()))]
etapas["Sem erros de coordenada"] = len(pres)

pres = pres[dentro_do_brasil(pres)]
etapas["Dentro do Brasil"] = len(pres)

pres = pres.drop_duplicates(["decimalLatitude", "decimalLongitude"])
etapas["Coordenadas únicas"] = len(pres)

linhas_p, colunas_p = rowcol(transform, pres.geometry.x.to_numpy(), pres.geometry.y.to_numpy())
pres["row"], pres["col"] = np.asarray(linhas_p), np.asarray(colunas_p)
pres = pres[valid[pres["row"], pres["col"]]]
etapas["Em pixel válido"] = len(pres)

pres = pres.drop_duplicates(["row", "col"])
etapas["Uma presença por pixel"] = len(pres)

display(pd.Series(etapas, name="Registros"))
```

A remoção de pixels inválidos precisa acontecer aqui. Na aula, ela só ocorria na modelagem (6906 amostras viraram 6827), o que quebra a proporção 1:1.

| Etapa (piloto) | Registros |
|---|---|
| Baixados do GBIF | 1599 |
| Com coordenada válida, sem (0,0) | 1599 |
| Sem erros de coordenada | 1429 |
| Dentro do Brasil | 1424 |
| Coordenadas únicas | 807 |
| Em pixel válido | 779 |
| Uma presença por pixel | **571** |

O resultado fica bem acima do mínimo sugerido de 100 presenças. As regras adicionais se justificam pela qualidade dos dados, não pelo AUC: com a limpeza mínima do enunciado (641 presenças), ele foi 0,972, contra 0,982 com a limpeza completa, uma diferença dentro da variação entre sementes. Elas existem para o modelo não aprender com coordenadas erradas.

## Etapa 4: pseudoausências

O `06/06` sorteia pontos no polígono do Brasil sem olhar os rasters, o que não atende ao enunciado. Aqui o sorteio é feito entre os pixels válidos, e o restante segue a aula: mesmo CRS métrico e mesma estrutura final do GeoPackage.

Escolhas para explicar no notebook:

- **Região:** todo o Brasil, como na aula.
- **Distância:** 20 km das presenças, o limite superior do enunciado, porque muitas coordenadas de herbário são imprecisas. Com 10 km, o AUC foi 0,953, contra 0,982 com 20 km. A diferença está dentro da variação entre sementes, mas um raio maior também tende a facilitar a separação das classes, então não use o AUC para escolher o raio.
- **Pixels:** o sorteio é sem reposição e o ponto fica no centro do pixel, então nenhum pixel se repete nem coincide com uma presença.
- **Proporção e semente:** 1:1 e `RANDOM_STATE = 123`.

```python
n_presencas = len(pres)
rng = np.random.default_rng(RANDOM_STATE)

# Sorteia pixels válidos a mais e descarta os que violam as restrições.
sorteados = rng.choice(np.flatnonzero(valid), size=n_presencas * 20, replace=False)
xs, ys = xy(transform, *np.unravel_index(sorteados, valid.shape))
candidatos = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs="EPSG:4326")
candidatos = candidatos[dentro_do_brasil(candidatos)].reset_index(drop=True)

perto, _ = pres.to_crs(METRIC_CRS).sindex.query(
    candidatos.to_crs(METRIC_CRS).geometry, predicate="dwithin", distance=BUFFER_KM * 1000
)
pseudo = candidatos.drop(index=np.unique(perto)).iloc[:n_presencas]

if len(pseudo) != n_presencas:
    raise RuntimeError("Candidatos insuficientes: aumente o fator de sorteio.")

samples = pd.concat(
    [pres[["geometry"]].assign(presence=1), pseudo[["geometry"]].assign(presence=0)],
    ignore_index=True,
)
samples = gpd.GeoDataFrame(samples, geometry="geometry", crs="EPSG:4326")
samples["id"] = np.arange(1, len(samples) + 1)
samples = samples[["id", "presence", "geometry"]]

samples_path = OUTPUT_DIR / "ErvaMate_pseudo_absence.gpkg"
samples.to_file(samples_path, driver="GPKG")
print(samples["presence"].value_counts())
```

Depois, copie a checagem visual do `06/06` (seção 12), definindo `output_path = samples_path` antes dela e trocando o título. As presenças devem se concentrar do Sul a Minas Gerais, com um grupo na Bahia. As pseudoausências devem cobrir o país inteiro.

## Etapa 5: modelagem

Siga o `06/08`, nesta ordem:

| Seção do `06/08` | O que fazer |
|---|---|
| 1 e 2 | **Substituir** pelo primeiro bloco abaixo |
| 3 a 6 | Copiar sem alterar |
| 7 | **Substituir** pelo segundo bloco abaixo |
| 8 a 14 | Copiar sem alterar |

```python
samples = gpd.read_file(samples_path)
samples["presence"] = samples["presence"].astype(int)
raster_names = [f.stem for f in raster_files]
```

Na seção 4, "Complete samples" deve ser igual ao total. Se cair, há um problema na máscara da Etapa 2.

**Por que substituir a seção 7.** Sem alterações, o notebook falha na seção 9 com `ValueError: Fold 7 validation contains only one class`, o mesmo erro salvo no notebook da aula. A mensagem engana: os folds 7 a 10 estão **vazios**. A pontuação da seção 7 favorece o fold mais próximo da meta de tamanho, então blocos pequenos continuam entrando em folds já cheios em vez de abrir um novo. No piloto, os folds de 1 a 6 ficaram com cerca de 150 amostras (meta: 91), e os demais, com nenhuma.

A `StratifiedGroupKFold` mantém cada bloco inteiro em um único fold e equilibra as classes, que é o objetivo declarado na própria seção 7. Registre essa troca no notebook. Como referência, a regra do `05/04` (cada bloco vai para o fold com menos amostras) também gera 10 folds válidos, mas com classes menos equilibradas: de 21 a 55 pseudoausências por fold, contra 44 a 50.

```python
from sklearn.model_selection import StratifiedGroupKFold

train_ids = samples_model["dataset"] == "Training"
blocks_train = samples_model.loc[train_ids, "block_id"].to_numpy()
y_folds = samples_df.loc[train_ids, "presence"].to_numpy()

sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
fold_lookup = {}
for fold, (_, val_idx) in enumerate(sgkf.split(blocks_train, y_folds, groups=blocks_train), start=1):
    fold_lookup.update(dict.fromkeys(blocks_train[val_idx], fold))

samples_model["fold"] = samples_model["block_id"].map(fold_lookup)
```

Depois da seção 14, acrescente esta comparação. Ela mostra quanto do AUC se deve apenas à geografia.

```python
auc_latitude = roc_auc_score(y_test, -test_spatial["latitude_cv"])
print("AUC do modelo:", round(roc_auc_score(y_test, prob_test), 3))
print("AUC só com a latitude:", round(auc_latitude, 3))
print("Blocos de teste com presença:", test_spatial.loc[y_test == 1, "block_id"].nunique())
```

| Medida (piloto) | Valor |
|---|---|
| Treino / teste | 914 / 228 amostras, 147 / 40 blocos |
| Folds | 10, com 39 a 70 presenças e 44 a 50 pseudoausências cada |
| Melhor combinação | `max_features="sqrt"`, `min_samples_leaf=5` |
| ROC AUC na validação espacial | 0,974 (média entre folds) |
| **ROC AUC no teste independente** | **0,982** |
| AUC só com a latitude | 0,968 |
| Blocos de teste com presença | 8 de 40 |
| Variáveis mais importantes | `bio1`, `bio11`, `bio9`, `bio5` e `bio10`, todas de temperatura, coerente com o clima Cfb citado pela Embrapa |

Como interpretar:

- **Não troque a semente para melhorar o AUC.** Ela decide quais blocos vão para o teste. No piloto, o AUC variou de 0,892 a 0,983 trocando a semente da divisão e de 0,929 a 0,994 trocando a das pseudoausências. Mantenha o valor fixado antes de ver o resultado e use o teste uma única vez, como dizem os slides da aula 05.
- **AUC alto não basta.** As presenças estão no Sul e no Sudeste, e as pseudoausências cobrem o país, então a latitude sozinha já separa boa parte das classes. Em uma das sementes, ela superou o modelo (0,981 contra 0,978). Relate o AUC junto com essa comparação.
- **Importância perto de zero não significa variável inútil.** 20 de 378 pares de preditores têm correlação acima de 0,9. Ao permutar uma variável, as correlacionadas compensam a perda. O maior valor foi 0,012 no piloto e 0,002 na Araucária.
- **A "probabilidade" é adequabilidade relativa.** Ela depende da proporção 1:1 escolhida, e o limiar de 0,50 é apenas a convenção do notebook.

## Etapa 6: mapas

Copie as seções 15 a 19 do `06/08`, trocando `Araucaria_RF_*` por `ErvaMate_RF_*` nos arquivos e "Araucaria" por "erva-mate" nos títulos. **Piloto:** cerca de 30 s com 12 núcleos.

O mapa esperado tem valores altos no Paraná, em Santa Catarina, no norte do Rio Grande do Sul, no sul de São Paulo e no sul de Mato Grosso do Sul. Uma faixa segue pelas serras de São Paulo, do Rio de Janeiro e de Minas Gerais até a Chapada Diamantina. Os valores são baixos no Pampa e quase zero na Amazônia. No piloto, 11,4 % dos pixels válidos ficaram como favoráveis no mapa binário.

A área principal coincide com a faixa de 21°S a 30°S do estudo da Embrapa. Ao norte dela, o mapa reflete os registros de Minas Gerais e da Bahia. Também aparecem manchas pequenas no Planalto Central, no sudoeste de Goiás, no Nordeste oriental e no extremo norte de Roraima. Só o entorno de Brasília tem presenças, então trate-as como áreas de clima parecido, não como ocorrência confirmada.

## Etapa 7: entrega

Esta célula final reúne os valores do formulário:

```python
resumo = {
    "Pontos antes da limpeza": etapas["Baixados do GBIF"],
    "Pontos depois da limpeza": etapas["Uma presença por pixel"],
    "Pseudoausências": int((samples["presence"] == 0).sum()),
    "ROC AUC do teste espacial independente": round(roc_auc_score(y_test, prob_test), 3),
    "Melhores hiperparâmetros": grid_search.best_params_,
}
display(pd.Series(resumo))
```

Checklist:

- [x] Apagar a pasta `output/` e rodar o notebook do zero, do download aos mapas. Feito em 30/10/2026, em um kernel novo. O download novo e o CSV já baixado deram os mesmos valores. Repita com **Restart e Run All** se mudar o notebook ou quiser atualizar os dados do GBIF.
- [ ] Revise as explicações do notebook sobre limpeza, pseudoausências, solo, folds e contorno do IBGE. O enunciado pede que as escolhas sejam explicadas.
- [ ] Confira a lista oficial de estados na Flora e Funga do Brasil (Etapa 3).
- [ ] Preencha o formulário com os valores do `resumo`. Eles precisam vir da última execução.
- [x] Montar `Matheus_ErvaMate.zip` só com `ervaMate.ipynb` e `output/ErvaMate_pseudo_absence.gpkg`, sem os rasters. Feito em 30/10/2026. Monte de novo se mudar o notebook ou rodá-lo outra vez.
- [ ] Envie pelo [formulário de entrega](https://docs.google.com/forms/d/e/1FAIpQLSdO5aJ_9MCZXcW42sDO97hwQVe5J1D184EtLDjYQgxYpWCFsg/viewform) até 18/11/2026, às 23h59. O upload exige login no Google.
