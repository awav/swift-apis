//  main.swift
//  Kalman
//
//  Created by Artem Artemev on 15/01/2020.
//

import TensorFlow
import QuartzCore


@differentiable
func kalman(
    observed: Tensor<Float>,
    likelihood: Tensor<Float>,
    prior: Tensor<Float>,
    length: Tensor<Float>
) -> Tensor<Float> {
    typealias T = Tensor<Float>
    let lam = sqrt(5.0) / length
    let lam1 = -3 * lam
    let lam2 = -(3 * 5) / length
    let lam3 = -pow(lam, 3)
    let z = T(0)
    let o = T(1)
    let F = T([T([z, o, z]), T([z, z, o]), T([lam3, lam2, lam1])])
    let H = T([T([o, z, z])])
    let kappa = (5.0 / 3.0) * prior / length.squared()
    let Pinf = T([T([prior,  z,     -kappa]),
                  T([z,      kappa, z]),
                  T([-kappa, z,     prior / pow(length, 4) * 25])])
    
    let A = F.squared()

    var m = T(zeros: [F.shape[0], 1])
    var P = Pinf
    var lml = z

    for k in 0..<observed.shape[0] {
        m = matmul(A, m)
        P = Pinf + matmul(A, matmul(P - Pinf, transposed: false, A, transposed: true))
        let S = likelihood + matmul(H, matmul(P, transposed: false, H, transposed: true))
        let v = observed[k] - matmul(H, m)
        let lz = -0.5 * (v.squared() / S + log(2 * Float.pi * S))
        let K = matmul(P, transposed: false, H, transposed: true) / S
        m = m + matmul(K, v)
        P = P - matmul(K, matmul(S, transposed: false, K, transposed: true))
        lml = lml + lz[0, 0]
    }

    return lml
}

func testKalmanFilter() {
    let num = 50000
    let y = Tensor<Float>(randomNormal: [num])
    let likelihood = Tensor<Float>(0.1)
    let prior = Tensor<Float>(1.0)
    let length = Tensor<Float>(1.0)

    kalman(observed: y, likelihood: likelihood, prior: prior, length: length)

//    gradient(at: likelihood, prior, length) {
//        kalman(observed: y, likelihood: $0, prior: $1, length: $2)
//    }
}

func executionTimeInterval(block: () -> ()) -> CFTimeInterval {
    print("Start execution")
    let start = CACurrentMediaTime()
    block()
    let end = CACurrentMediaTime()
    print("Finished execution")
    return end - start
}


let time = executionTimeInterval {
    testKalmanFilter()
}

print("Elapsed time: \(time)")
