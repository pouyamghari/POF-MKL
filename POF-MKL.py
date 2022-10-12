import numpy as np
from numpy import linalg as LA

class POFMKL:
    def __init__(self, lam, rf_feature, eta, M):
        self.lam = lam
        self.rf_feature = np.array(rf_feature)
        self.eta = eta
        self.M = M
        self.theta = np.zeros((2*rf_feature.shape[1],rf_feature.shape[2]))
        self.num_kernels = rf_feature.shape[2]
        
    def predict(self, X, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        f_RF_p = np.zeros((b,1))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(self.theta[:,j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def local_update(self, f_RF_p, Y, w, X_features, selected_kernels, prob):
        b, n_components = X_features.shape
        l = np.zeros((1,self.num_kernels))
        local_grad = np.zeros((n_components, self.M))
        for j in range(self.num_kernels):
            l[0,j] = (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(self.theta[:,j])**2)
            w[0,j] = w[0,j]*np.exp(-self.eta*l[0,j])
        for i, j in enumerate(selected_kernels):
            local_grad[:,i] = (self.eta/prob)*( (2*(f_RF_p[j,0] -Y)*np.transpose(X_features[j,:]))+2*self.lam*self.theta[:,j] )
        return w, local_grad
    
    def kernel_selection(self, w, zeta):
        sorted_arg_w = np.argsort(w[0,:])[::-1]
        u = np.zeros((1,int(np.ceil(self.num_kernels/self.M))))
        for i in range(int(np.ceil(self.num_kernels/self.M))):
            u[0,i] = np.sum( w[0,sorted_arg_w[i*self.M:min((i+1)*self.M,self.num_kernels)]] )
        u/=np.sum(u)
        u = (1-zeta)*u + (zeta/np.ceil(self.num_kernels/self.M))*np.ones((1,int(np.ceil(self.num_kernels/self.M))))
        r = np.random.rand()
        I = 0
        u_sum = np.cumsum(u)
        while r>u_sum[I]:
            I+=1
        selected_kernels = sorted_arg_w[I*self.M:min((I+1)*self.M,self.num_kernels)]
        prob = u[0,I]
        return selected_kernels, prob
            
    def global_update(self, agg_grad, agg_kernel_indices):
        theta_update = np.zeros(self.theta.shape)
        for i, selected_kernels in enumerate(agg_kernel_indices):
            for j, ind_kernel in enumerate(selected_kernels):
                theta_update[:,ind_kernel]+=agg_grad[i][:,j]
        for i in range(self.num_kernels):
            self.theta[:,i]-=(theta_update[:,i]/len(agg_grad))