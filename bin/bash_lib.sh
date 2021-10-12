# a library of helper functions for scripting
# shell scripts should source this library.

# the highest level commands in here are:  run_gpus round_robbin_gpu run fork
# the lockfile function has very useful environment variable options.

function _lockfile_use_flock_if_atomic() {
  # internal function.  obtain a lock on the lockfile_fp if atomic=true
  local fd="$1"
  local atomic="$2"
  local cmd;
  if [ "${atomic}" = true ] ; then
    cmd="flock -w 5 $fd"  # || exit 1
  elif [ "${atomic}" = false ] ; then
    cmd="/bin/true"  # do nothing
  else
    echo "ERROR: _lockfile_set_stats requires atomic=[true|false]"
    return 1
  fi
  echo $cmd
}
export -f _lockfile_use_flock_if_atomic

function _lockfile_get_tmpdir() {
  # internal function used to record num_running jobs
  echo "/tmp/lockfile/$(cksum <<< "$(realpath -m ${1})" | cut -f1 -d\ )"
}
export -f _lockfile_get_tmpdir

function _lockfile_get_stats() {
  # helper to get the num_successes, num_failures and num_running jobs.
  # usage:  read num_successes num_failures num_running < <(_lockfile_get_stats "$lockfile_fp" is_atomic )
  # where is_atomic is true or false.  if true, obtain lock on lockfile_fp before getting stats
  local lockfile_fp="$1"
  local atomic="${2:-true}"
  touch "$lockfile_fp"
  {
    $(_lockfile_use_flock_if_atomic 9 $atomic)
    local num_successes="$( (grep success_count\= $lockfile_fp || echo 0=0) | cut -f2 -d\= )"
    local num_failures="$(  (grep failure_count\= $lockfile_fp || echo 0=0) | cut -f2 -d\= )"
    local _tmpdir="$(_lockfile_get_tmpdir "$lockfile_fp")"
    local num_running="$(find "$_tmpdir" -type f 2>/dev/null | wc -l)"
    echo $num_successes $num_failures $num_running  # this is a return value
    } 9<"$lockfile_fp"
}
export -f _lockfile_get_stats

function _lockfile_increment_stats() {
  # Increment stats using given integer deltas.
  # usage: _lockfile_increment_stats lockfile_fp delta_success delta_fail atomic
  #
  #If atomic=true, obtain and wait for a lock on the lockfile_fp.
  local lockfile_fp="$1"
  local atomic="${4:-true}"
  touch "$lockfile_fp"
  {
  $(_lockfile_use_flock_if_atomic 11 $atomic)
  read _num_successes _num_failures _num_running < <(_lockfile_get_stats "$lockfile_fp" false )
  local num_successes=$(($_num_successes+$2))
  local num_failures=$(($_num_failures+$3))
  _lockfile_set_stats "$lockfile_fp" $num_successes $num_failures false
  } 11<"$lockfile_fp"
}
export -f _lockfile_increment_stats

function _lockfile_set_stats() {
  # Set the number of successes and failures for a given lockfile
  # usage: _lockfile_set_stats lockfile_fp num_successes num_failures atomic
  #
  # where atomic=[true|false].  If true, obtain and wait for a lock on lockfile_fp before continuing
  local lockfile_fp="$1"
  local num_successes="$2"
  local num_failures="$3"
  local atomic="${4:-true}"
  touch "$lockfile_fp"
  {
  $(_lockfile_use_flock_if_atomic 10 $atomic)
  cat <<EOF >$lockfile_fp
local success_count=$num_successes
local failure_count=$num_failures
EOF
  } 10<"$lockfile_fp"
}
export -f _lockfile_set_stats

function _lockfile_reset_running_count() {
  local lockfile_fp="$1"
  local atomic="${2:-true}"
  touch $lockfile_fp
  {
  $(_lockfile_use_flock_if_atomic 12 $atomic)
  local tmpdir="$(_lockfile_get_tmpdir "$lockfile_fp")"
  case $tmpdir in /tmp/lockfile/*)  # sanity check that rm -rf is right place
    rm -rf $tmpdir
  esac
  } 12<"$lockfile_fp"
}
export -f _lockfile_reset_running_count

# bash traps to handle graceful exits
function _lockfile_trap_exit() {
  rc=$?
  local lockfile_fp="$1"
  local __lockfile_active_job_fp="$2"
  # echo "TRAP EXIT $__lockfile_active_job_fp" $rc
  if grep -q failed $__lockfile_active_job_fp ; then
  # if [ $rc -ne 0 ] ; then
    rc=$rc
    local state='fail'
    _lockfile_increment_stats "$lockfile_fp" 0 1 true
    local RED='\033[0;31m'
    local NC='\033[0m' # No Color
    echo -e "${RED} FAILURE  ${NC} lockfile_fp= ${lockfile_fp}"
  else
    rc=0
    local GREEN='\033[0;32m'
    local NC='\033[0m' # No Color
    echo -e "${GREEN} Success! ${NC} lockfile_fp= ${lockfile_fp} "
    local state='success'
    _lockfile_increment_stats "$lockfile_fp" 1 0 true
  fi
  echo -e "STATE=$state\tDATE=\"$(date)\"\tHOSTNAME=\"$(hostname)\"" >> ${lockfile_fp}.log
  # remove the counter for the current job
  rm $__lockfile_active_job_fp
  # try to remove the tmp directory for active jobs associated with this
  # lock in case no jobs are actively running.
  rmdir $(_lockfile_get_tmpdir "$lockfile_fp") 2>/dev/null || true
}
export -f _lockfile_trap_exit
function _lockfile_trap_err() {
  local __lockfile_active_job_fp="$1"
  # echo "TRAP ERR $__lockfile_active_job_fp"
  echo failed > $__lockfile_active_job_fp  # bubbles to trap_exit
}
export -f _lockfile_trap_err
function trap_many_fns() {
  local newcmd=$1
  local signal=$2
  local cmd;
  cmd="$(trap -p $signal)"
  cmd="${cmd#*\'}"  # strip from left side the phrase "trap -- '"
  cmd="${cmd%\'*}"  # strip from right side the "' SIGNAL"
  if [ -n "$cmd" ] ; then  # trap is defined
    cmd="${cmd} ; $newcmd"
  else
    cmd="$newcmd"
  fi
  trap "$cmd" $signal
}
export -f trap_many_fns

function _lockfile_reset() {
  # helper function to clear all stats.  Useful for manually testing lockfile function.
  # usage:  _lockfile_reset filename_for_the_lock
  local lockfile_fp="$1"
  _lockfile_set_stats "$lockfile_fp" 0 0
  _lockfile_reset_running_count "$lockfile_fp"
}
export -f _lockfile_reset

# Ensure only N jobs complete, with at most M errors by making use of atomic
# writes to a given file on disk with the linux "flock" program.
# You can safely run jobs in parallel on this machine with lock guarantees.
#
# usage:
#    {
#      get_lockfile filename_for_the_lock || exit 1
#      echo "my code here" ; sleep 2  # for example (sleep is not necessary)
#    }
#
# Environment variable options:
#     lockfile_maxsuccesses=N  Run a command N times, for any positive integer N
#     lockfile_maxfailures=M  Attempt to run a command up to M failures
#     lockfile_maxconcurrent=O  Let at most O concurrent players access the lock.  Counts files in /tmp/lockfile_XXX/ to find the current active jobs.
#     lockfile_ignore=true  Define this to completely bypass the lockfile function  (i.e. do nothing)
function lockfile() {
  if [ "${lockfile_ignore:-false}" = true ] ; then
    echo "bypassing lockfile"  # just ignore lockfile
    return 0
  fi
  local lockfile_fp="$(realpath -m ${1})"
  local lockfile_maxsuccesses="${lockfile_maxsuccesses:-1}"
  local lockfile_maxfailures="${lockfile_maxfailures:-1}"
  local lockfile_maxconcurrent="${lockfile_maxconcurrent:-1}"

  # enable getting / setting the lock
  touch $lockfile_fp
  local _tmpdir="$(_lockfile_get_tmpdir "$lockfile_fp")"
  mkdir -p "$_tmpdir"

  # Try to obtain the lock, or exit.  Everything inside the block is atomic.
  # The return code determines if obtained the lockfile (exit 0) vs didn't get it (exit non-zero)
  {
    set -eE
    set -u
    set -o pipefail
    set -o errtrace 
    flock -w 1 9 || return 1
    read num_successes num_failures num_running < <(_lockfile_get_stats "$lockfile_fp" false)

    # sanity checks to verify can obtain lock
    if [ "$lockfile_maxsuccesses" -le "$num_successes" ] ; then
      local YELLOW='\033[0;33m'
      local NC='\033[0m' # No Color
      echo -e "$YELLOW Job previously completed $num_successes times.  Not running.  $NC lockfile_fp= $lockfile_fp"
      exit 2
    fi
    if [ "$lockfile_maxconcurrent" -le "$num_running" -o "$lockfile_maxsuccesses" -le $(($num_successes+$num_running)) ] ; then
      local YELLOW='\033[0;33m'
      local NC='\033[0m' # No Color
      echo $lockfile_maxconcurrent $num_running $lockfile_maxsuccesses $(($num_successes+$num_running))
      echo -e "$YELLOW Sufficient number of currently active jobs ($num_running running + $num_successes previous successes == $(($num_successes+$num_running)) ).  Not running.  If this is incorrect, remove tmpfile.  It was probably caused by abrupt shutdown.  $NC lockfile_fp= $lockfile_fp  tmpfile= $(_lockfile_get_tmpdir $lockfile_fp)"
      exit 3
    fi
    if [ "$lockfile_maxfailures" -le "$num_failures" ] ; then
      local RED='\033[0;31m'
      local NC='\033[0m' # No Color
      echo -e "$RED Too many previous failures ($lockfile_maxfailures) reached. Not running. You can set lockfile_maxfailures=N or lockfile_ignore=true. $NC lockfile_fp= $lockfile_fp"
      exit 4
    fi

    # increment count of currently running jobs using a temporary file (ephemeral storage)
    local __lockfile_active_job_fp="$(mktemp -p "$_tmpdir")"
    date '+%Y-%m-%dT%H:%M:%S.%N' >> $__lockfile_active_job_fp
    echo "$lockfile_fp" >> $__lockfile_active_job_fp

    # set up bash traps (these will append to existing traps)
    trap_many_fns "_lockfile_trap_exit ""$lockfile_fp"" ""$__lockfile_active_job_fp" EXIT # always run this
    trap_many_fns "_lockfile_trap_err ""$__lockfile_active_job_fp" INT
    trap_many_fns "_lockfile_trap_err ""$__lockfile_active_job_fp" ERR
  } 9<"$lockfile_fp"
  rc=$?  # we got the lock if rc=0
  if [ $rc -eq 2 -o $rc -eq 3 -o $rc -eq 4 ] ; then
    return $rc
  elif [ $rc -eq 1 ] ; then
    echo "Failed to obtain lock.  Too many competing processes"  $lockfile_fp
    return $rc
  elif [ $rc -eq 0 ] ; then
    # we got the lock!  let user run code
    return 0
  else
    echo "CODE BUG: unrecognized exit code in lockfile()."
    return $rc
  fi
}
export -f lockfile


function log_initial_msgs() {(
  set -eE
  set -u
  local run_id=$1
  echo "Running on hostname: $(hostname)"
  echo "run_id: ${run_id}"
  date

  # print out current configuration
  echo ======================
  echo CURRENT GIT CONFIGURATION:
  echo "git commit: $(git rev-parse HEAD)"
  echo
  echo git status:
  git status
  echo
  echo git diff:
  git --no-pager diff --cached
  git --no-pager diff
  echo
  echo ======================
  echo
  echo
)}
export -f log_initial_msgs


# run command and log stdout/stderr
function run() {
  local run_id="$1"
  shift
  local cmd="$*"
  local timestamp="$(date +%Y%m%dT%H%M%S.%N)"
  local lockfile_path="./results/$run_id/lock"
  local log_fp="./results/$run_id/${timestamp}_console.log"
  local gitlog_fp="./results/$run_id/${timestamp}_git.log"

  mkdir -p "$(dirname "$(realpath -m "$log_fp")")"
  (
    set -eE
    set -u
    set -o pipefail
    lockfile "${lockfile_path}"
    echo "START: $timestamp"
    (log_initial_msgs "$run_id" >$gitlog_fp 2>&1)
    echo run_id="$run_id" "$cmd"
    # set +eE
    # set +o pipefail
    # set +u
    run_id="$run_id" $cmd
    rc=$?
    # set -eE
    # set -o pipefail
    local RED='\033[0;31m'
    local GREEN='\033[0;32m'
    local NC='\033[0m' # No Color
    if [ $rc = 0 ] ; then
      local color="$GREEN"
    else 
      local color="$RED"
    fi
    echo -e "${color}END: $(date +%Y%m%dT%H%M%S.%N) \t EXIT_CODE: $rc $NC"
  ) 2>&1 | tee "$log_fp" 2>&1
}
export -f run


# run jobs with logging of stdout.  useful in conjunction with wait.
#  > fork experiment_name1  some_command
#  > fork experiment_name2  another_command
#  > wait
function fork() {
  (run $@) &
}
export -f fork



function round_robbin_gpu() {
  # distribute `num_tasks` tasks on each of the (locally) available gpus

  # in round robbin fashion.  This implementation is synchronized; a set of
  # `num_tasks` tasks must complete before another set of `num_tasks` tasks
  # starts.

  # NOTE: use `run_gpus` instead if you want one task per gpu, as it doesn't block.

  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local num_tasks=${1:-$num_gpus}  # how many concurrent tasks per gpu
  local idx=0

  while read -r line0 ; do

    local gpu_idx=$(( $idx % num_gpus ))
    CUDA_VISIBLE_DEVICES=$gpu_idx device=cuda:$gpu_idx fork $line0
    local idx=$(( ($idx + 1) % $num_tasks ))
    if [ $idx = 0 ] ; then
      wait # ; sleep 5
    fi
  done
  if [ $idx != 0 ] ; then
    wait # ; sleep 5
  fi
}
export -f round_robbin_gpu


function run_gpus() {
  # Run a set of tasks, one task per gpu, by populating and consuming from a Redis queue.

  # use redis database as a queuing mechanism.  you can specify how to connect to redis with RUN_GPUS_REDIS_CLI 
  local num_tasks_per_gpu="${1:-1}"
  local redis="${RUN_GPUS_REDIS_CLI:-redis-cli -n 1}"
  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local Q="`mktemp -u -p run_gpus`"

  # trap "$(echo $redis DEL "$Q" "$Q/numstarted") > /dev/null" EXIT

  # --> publish to the redis queue
  local maxjobs=0
  while read -r line0 ; do
    $redis --raw <<EOF > /dev/null
MULTI
LPUSH $Q "$line0"
EXPIRE "$Q" 1209600
EXEC
EOF
    # $redis LPUSH "$Q" "$line0" >/dev/null
    local maxjobs=$(( $maxjobs + 1 ))
  done
  # --> start the consumers
  for gpu_idx in `nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u` ; do
    for i in $(seq 1 $num_tasks_per_gpu) ; do
      # TODO: not ideal that we have multiple consumers on same queue because they're all competing for same elements
      # in future, should have additional consumers randomly sample from the queue
      # or have a better queuing mechanism
      sleep $(bc -l <<< "scale=4 ; ${RANDOM}/32767/10")
      consumergpu_redis $gpu_idx "$redis" "$Q" $maxjobs &
  done ; done
  wait
  $redis DEL "$Q" "$Q/numstarted" >/dev/null
}


function consumergpu_redis() {
  local gpu_idx=$1
  local redis="$2"
  local Q="$3"
  local maxjobs=$4
  local num_started=0

  while [ $num_started -lt $maxjobs ] ; do
    # --> query redis for a job to run
    local redisrv="$($redis --raw <<EOF
MULTI
LMOVE $Q $Q RIGHT LEFT
INCR $Q/numstarted
EXPIRE "$Q" 1209600
EXPIRE "$Q/numstarted" 1209600
EXEC
EOF
)"
    local cmd2="$( echo "$redisrv" | head -n 6 | tail -n 1)"
    local num_started="$( echo "$redisrv" | head -n 7 | tail -n 1)"
    CUDA_VISIBLE_DEVICES=$gpu_idx run $cmd2
    done
}
export -f run_gpus
