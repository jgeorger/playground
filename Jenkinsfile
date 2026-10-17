pipeline {
    // Needs a node with the CUDA toolkit, CMake 3.28+, Ninja, a GPU, and MathDx
    // at the path hard-coded in CMakeLists.txt. The Jenkins controller container
    // has none of these, so this runs on an agent on the GPU host.
    agent { label 'cuda' }

    options {
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    environment {
        PATH = "/usr/local/cuda/bin:${env.PATH}"
    }

    stages {
        stage('Configure') {
            steps {
                sh 'cmake --preset release'
            }
        }

        stage('Build') {
            steps {
                sh 'cmake --build --preset release'
            }
        }

        stage('Test') {
            steps {
                sh 'rm -rf test-results && mkdir -p test-results'
                script {
                    // GoogleTest suites write JUnit XML for per-case reporting
                    def gtests = ['test_segment_merger', 'test_identity_batched', 'test_bpdn']
                    // Plain executables that return nonzero on failure
                    def plainTests = ['test_linear_solve_dx']

                    // catchError lets every test run even if an earlier one fails
                    gtests.each { t ->
                        catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                            sh "./build/release/${t} --gtest_output=xml:test-results/${t}.xml"
                        }
                    }
                    plainTests.each { t ->
                        catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                            sh "./build/release/${t}"
                        }
                    }
                }
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results/*.xml'
        }
    }
}
