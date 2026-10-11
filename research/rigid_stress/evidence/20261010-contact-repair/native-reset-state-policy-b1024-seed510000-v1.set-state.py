@mutates(StateChange.GEOMETRY, StateChange.DYNAMICS)
def set_state(self, f, state, envs_idx=None, *, partial: bool=False) -> None:
    if not self.is_active:
        return
    if partial:
        self.collider.reset(envs_idx)
        self.constraint_solver.reset(envs_idx)
    else:
        self.collider.clear(envs_idx)
        self.constraint_solver.clear(envs_idx)
    if False:
        errno = qd_to_torch(self._errno, copy=False)
        qpos_dst = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
        vel_dst = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
        acc_dst = qd_to_torch(self.dyn_state.dofs.acc, transpose=True, copy=False)
        ctrl_force_dst = qd_to_torch(self.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
        ctrl_mode_dst = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
        pos_dst = qd_to_torch(self.dyn_state.links.pos, transpose=True, copy=False)
        quat_dst = qd_to_torch(self.dyn_state.links.quat, transpose=True, copy=False)
        cfrc_vel_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_vel, transpose=True, copy=False)
        cfrc_ang_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_ang, transpose=True, copy=False)
        fric_dst = qd_to_torch(self.dyn_state.geoms.friction_ratio, transpose=True, copy=False)
        if self._use_hibernation:
            links_hibernated_dst = qd_to_torch(self.dyn_state.links.is_hibernated, transpose=True, copy=False)
            awake_steps_dst = qd_to_torch(self.dyn_state.links.awake_steps, transpose=True, copy=False)
            dofs_hibernated_dst = qd_to_torch(self.dyn_state.dofs.is_hibernated, transpose=True, copy=False)
            geoms_hibernated_dst = qd_to_torch(self.dyn_state.geoms.is_hibernated, transpose=True, copy=False)
            entities_hibernated_dst = qd_to_torch(self.dyn_state.entities.is_hibernated, transpose=True, copy=False)
            islands_hibernated_dst = qd_to_torch(self.constraint_solver.constraint_state.island.is_hibernated, transpose=True, copy=False)
            islands_next_link_dst = qd_to_torch(self.constraint_solver.constraint_state.island.hibernated_next_link, transpose=True, copy=False)
            n_awake_dofs_dst = qd_to_torch(self.rigid_info.n_awake_dofs, copy=False)
        if envs_idx is not None and (not isinstance(envs_idx, torch.Tensor)):
            (envs_idx,) = indices_to_mask(envs_idx)
        if isinstance(envs_idx, torch.Tensor):
            if envs_idx.dtype == torch.bool:
                envs_mask = envs_idx
            else:
                envs_mask = torch.zeros(self._B, dtype=torch.bool, device=gs.device)
                envs_mask[envs_idx] = True
            errno.masked_fill_(envs_mask, 0)
            if self.n_qs:
                torch.where(envs_mask[:, None], state.qpos, qpos_dst, out=qpos_dst)
                torch.where(envs_mask[:, None], state.dofs_vel, vel_dst, out=vel_dst)
                torch.where(envs_mask[:, None], state.dofs_acc, acc_dst, out=acc_dst)
                ctrl_force_dst.masked_fill_(envs_mask[:, None], 0.0)
                ctrl_mode_dst.masked_fill_(envs_mask[:, None], gs.CTRL_MODE.FORCE)
            torch.where(envs_mask[:, None, None], state.links_pos, pos_dst, out=pos_dst)
            torch.where(envs_mask[:, None, None], state.links_quat, quat_dst, out=quat_dst)
            cfrc_vel_dst.masked_fill_(envs_mask[:, None, None], 0.0)
            cfrc_ang_dst.masked_fill_(envs_mask[:, None, None], 0.0)
            if self.n_geoms:
                torch.where(envs_mask[:, None], state.friction_ratio, fric_dst, out=fric_dst)
            if self._use_hibernation:
                links_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                awake_steps_dst.masked_fill_(envs_mask[:, None], 0)
                dofs_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                geoms_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                entities_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                islands_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                islands_next_link_dst.masked_fill_(envs_mask[:, None], -1)
                n_awake_dofs_dst.masked_fill_(envs_mask, self.n_dofs)
        else:
            if self.n_qs:
                errno[envs_idx] = 0
                qpos_dst[envs_idx] = state.qpos[envs_idx]
                vel_dst[envs_idx] = state.dofs_vel[envs_idx]
                acc_dst[envs_idx] = state.dofs_acc[envs_idx]
                ctrl_force_dst[envs_idx] = 0.0
                ctrl_mode_dst[envs_idx] = gs.CTRL_MODE.FORCE
            pos_dst[envs_idx] = state.links_pos[envs_idx]
            quat_dst[envs_idx] = state.links_quat[envs_idx]
            cfrc_vel_dst[envs_idx] = 0.0
            cfrc_ang_dst[envs_idx] = 0.0
            if self.n_geoms:
                fric_dst[envs_idx] = state.friction_ratio[envs_idx]
            if self._use_hibernation:
                links_hibernated_dst[envs_idx] = 0
                awake_steps_dst[envs_idx] = 0
                dofs_hibernated_dst[envs_idx] = 0
                geoms_hibernated_dst[envs_idx] = 0
                entities_hibernated_dst[envs_idx] = 0
                islands_hibernated_dst[envs_idx] = 0
                islands_next_link_dst[envs_idx] = -1
                n_awake_dofs_dst[envs_idx] = self.n_dofs
        if gs.backend == gs.metal:
            torch.mps.synchronize()
    else:
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        kernel_set_zero(envs_idx, self._errno)
        kernel_set_state(envs_idx, state.qpos, state.dofs_vel, state.dofs_acc, state.links_pos, state.links_quat, state.friction_ratio, self.dyn_state, self.rigid_info, self.rigid_config)
        if self._use_hibernation:
            kernel_reset_hibernation(envs_idx, self.dyn_state, self.constraint_solver.constraint_state, self.dyn_info, self.rigid_info, self.rigid_config)
    if not partial:
        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        if envs_idx.dtype == torch.bool:
            fn = kernel_masked_forward_kinematics_links_geoms
        else:
            fn = kernel_forward_kinematics_links_geoms
        fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True
    else:
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False
    self._restart()
