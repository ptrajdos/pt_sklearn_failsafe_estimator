ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))


SRCDIR=${ROOTDIR}/pt_sklearn_failsafe_estimator
TESTDIR=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
REQ_FILE=${ROOTDIR}/requirements_dev.txt
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
DOCS_DIR=${ROOTDIR}/docs

COVERAGE = coverage
COVERAGERC=${ROOTDIR}/.coveragerc
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYTHON=python
PIP=pip

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log


ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean test docs

all: venv

venv:
	${PYTHON} -m venv ${VENV_SUBDIR}
	${ACTIVATE}; ${PIP} install -e ${ROOTDIR} --prefer-binary --log ${INSTALL_LOG_FILE}; ${PIP} install --prefer-binary -r ${REQ_FILE};

test: venv
	mkdir -p ${LOGDIR}  
	${ACTIVATE}; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	${ACTIVATE}; ${COVERAGE} html --show-contexts

test_parallel: venv
	mkdir -p ${COVDIR} ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --class-fixtures -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ${COVERAGERC} --coverage-source ${SRCDIR} --coverage-html ${COVDIR}  2>&1 |tee -a ${LOGFILE}

docs:
	mkdir -p ${DOCS_DIR}
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

clean_venv:
	rm -rf ${VENV_SUBDIR}
