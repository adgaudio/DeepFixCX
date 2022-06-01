# code to download and extract datasets
function kim_eye() {
  # glaucoma dataset
  # https://doi.org/10.1371/journal.pone.0207982
  # https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/1YRRAC
  wget \
     "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/1YRRAC/OGRSQO" \
     -O kim_eye.zip --continue
  if ! md5sum --status -c <(echo 305f29a31a7db2c9bd5536b9c077a09e kim_eye.zip) ; then
    echo "ERROR: kim_eye: invalid md5sum"
    return 1
  fi
  mkdir kim_eye || true
  pushd kim_eye
  unzip ../kim_eye.zip

  popd
  chmod -R -w ./kim_eye  # protect the dataset
}

function hrf() {
  # glaucoma dataset
  wget \
    "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip" \
    -O hrf.zip --continue
  mkdir hrf || true
  pushd hrf
  unzip ../hrf.zip
  popd
  chmod -R -w ./hrf  # protect the dataset
}

function acrima() {
  # glaucoma dataset
  # note: there are other files available from the dataset.
  # these are large files to reproduce the bootstrap experiments.
  local fps=("acrima_database.zip" "acrima_trained_models.zip")
  local urls=(
    "https://figshare.com/ndownloader/files/14137700?private_link=c2d31f850af14c5b5232"
    "https://figshare.com/ndownloader/files/14137712?private_link=c2d31f850af14c5b5232"
  )
  local md5=("75c39a5d731ea17374396589e3273c5f" "831d15f28446883786f26b695c4cb1cb")

  mkdir acrima || true

  for idx in 0 1 ; do
    local fp="${fps[$idx]}"
    wget \
      "${urls[$idx]}" \
      -O "$fp" --continue
    if ! md5sum --status -c <(echo "${md5[$idx]}" "$fp" ) ; then
      echo "ERROR: $fp: invalid md5sum"
      return 1
    fi
    pushd acrima
    unzip ../$fp
    popd
  done
  chmod -R -w ./acrima/Database  # protect the dataset
}


set -e
set -u

# cd into repo root
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# make data directory
mkdir ./data || true
cd ./data

# download and extract datasets
kim_eye
# acrima
# hrf
