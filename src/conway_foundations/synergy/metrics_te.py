"""Transfer entropy: directional information flow."""

import numpy as np

from conway_foundations.synergy.metrics import register_metric, trajectory_outcomes


class TransferEntropyMetric:
    """TE(X→Y): how much X's past predicts Y's future beyond Y's own past."""

    name = "transfer_entropy"

    def __call__(self, g1, g2, trajectory):
        traj = list(trajectory)
        x = trajectory_outcomes(g1, traj)
        y = trajectory_outcomes(g2, traj)

        if len(x) < 3:
            return {
                "te_x_to_y": 0.0,
                "te_y_to_x": 0.0,
                "te_symmetric": 0.0,
                "te_asymmetry": 0.0,
                "sample_size": float(len(x)),
            }

        te_xy = self._compute_te(x[:-1], y[:-1], y[1:])
        te_yx = self._compute_te(y[:-1], x[:-1], x[1:])

        return {
            "te_x_to_y": te_xy,
            "te_y_to_x": te_yx,
            "te_symmetric": (te_xy + te_yx) / 2,
            "te_asymmetry": abs(te_xy - te_yx),
            "sample_size": float(len(x)),
        }

    def _compute_te(self, x_past, y_past, y_next):
        """TE(X→Y) = I(Y_next; X_past | Y_past) via empirical counts."""
        n = len(x_past)
        joint_count: dict[tuple[int, int, int], int] = {}
        yp_xp_count: dict[tuple[int, int], int] = {}
        yn_yp_count: dict[tuple[int, int], int] = {}
        yp_count: dict[int, int] = {}

        for i in range(n):
            xp, yp, yn = int(x_past[i]), int(y_past[i]), int(y_next[i])
            joint_count[(yn, yp, xp)] = joint_count.get((yn, yp, xp), 0) + 1
            yp_xp_count[(yp, xp)] = yp_xp_count.get((yp, xp), 0) + 1
            yn_yp_count[(yn, yp)] = yn_yp_count.get((yn, yp), 0) + 1
            yp_count[yp] = yp_count.get(yp, 0) + 1

        te = 0.0
        for (yn, yp, xp), c_joint in joint_count.items():
            p_joint = c_joint / n
            p_yn_given_yp_xp = c_joint / yp_xp_count[(yp, xp)]
            p_yn_given_yp = yn_yp_count[(yn, yp)] / yp_count[yp]
            if p_yn_given_yp_xp > 0 and p_yn_given_yp > 0:
                te += p_joint * np.log2(p_yn_given_yp_xp / p_yn_given_yp)

        return max(0.0, te)


register_metric(TransferEntropyMetric())
