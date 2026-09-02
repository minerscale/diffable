use num_traits::Zero;

use crate::{
    hypersphere::So3,
    quaternion::Quaternion,
    traits::{
        Euclidean, Field, Quotient, SemidirectJetStructure, SemidirectProduct, SemidirectStructure,
        Tensor,
        calculus::{CommutesJet, DirectSum, JetVector, JetVectorIn, Tangent},
        𝐑𝐞𝐚𝐥,
    },
};

#[derive(Debug, Copy, Clone)]
pub struct RigidMotionStructure;

pub type Se3<V> = SemidirectProduct<V, V, So3<V>, V, RigidMotionStructure>;

impl<V: Euclidean> SemidirectStructure<V, V, So3<V>, V> for RigidMotionStructure {
    fn alpha<const K: usize>(g: Tangent<So3<V>, V, K>, n: Tangent<V, V, K>) -> Tangent<V, V, K> {
        const { assert!(V::N == 3) }

        let g = So3::<JetVectorIn<_, V, K>>::commute_jet(g);
        let n = n.into_jet(|value| value).retag();
        let q = g.lift().to_quaternion();
        let n = Quaternion::new(Zero::zero(), n[0], n[1], n[2]);

        let acted = q * n * q.conj();

        JetVector::<V, K>::from_fn(|coordinate| acted[coordinate + 1].retag())
            .into_tangent(|value| value)
    }

    fn identity_exp<const K: usize>(
        coordinate: JetVector<DirectSum<V, V>, K>,
    ) -> Tangent<Se3<V>, DirectSum<V, V>, K> {
        Se3::<V>::abelian_identity_exp_series::<K, 24>(coordinate)
    }

    fn identity_log<const K: usize>(
        point: Tangent<Se3<V>, DirectSum<V, V>, K>,
    ) -> Option<JetVector<DirectSum<V, V>, K>> {
        Se3::<V>::abelian_identity_log_series::<K, 24, 24>(point)
    }
}

impl<V: Euclidean> SemidirectJetStructure<V, V, So3<V>, V> for RigidMotionStructure {
    type JetCategory = 𝐑𝐞𝐚𝐥::𝒞;
    type JetG<const K: usize> = So3<JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, K>>;
    type JetN<const K: usize> = JetVectorIn<𝐑𝐞𝐚𝐥::𝒞, V, K>;
    type JetStructure<const K: usize> = RigidMotionStructure;
}
