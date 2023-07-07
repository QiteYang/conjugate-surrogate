classdef CS_KRVEA< ALGORITHM
% <multi/many> <real/integer> <expensive>
% Surrogate-assisted RVEA
% alpha ---  2 --- The parameter controlling the rate of change of penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, A surrogate-
% assisted reference vector guided evolutionary algorithm for
% computationally expensive many-objective optimization, IEEE Transactions
% on Evolutionary Computation, 2018, 22(1): 129-142.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha,wmax,mu] = Algorithm.ParameterSet(2,20,5);
            M = Problem.M;
            D = Problem.D;

            %% Generate the reference points and population
            [V0,Problem.N] = UniformPoint(Problem.N,Problem.M);
            V     = V0;
            NI    = 11*Problem.D-1;
            P     = UniformPoint(NI,Problem.D,'Latin');
            A2    = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            A1    = A2;  
            %% Model
            THETA = 5.*ones(M,D);
            Model = cell(1,M);
            THETA_conj = 5.*ones(M,D+M-1);
            Model_conj = cell(1,M);
            

            %% Optimization
            while Algorithm.NotTerminated(A2)
                % Refresh the model and generate promising solutions
                A1Dec = A1.decs;
                A1Obj = A1.objs;
                joint = 1:1:D;
                Rele_var = cell(1,M);
                
                % 根据相关性确定每个目标的relevant variables
                for i = 1 : M
                    [idx,KN_point,weights] = featureSelect(A1Dec,A1Obj(:,i));
                    [~,Promising] = find(weights>weights(KN_point));
                    if length(Promising) < ceil(0.5*D)
                        [~,Promising] = find(idx<=ceil(0.5*D));
                    end
                    Rele_var{i} = Promising;
                    joint = intersect(joint,Promising); % 公共relevant variables
                end
                
                % 降维后的模型，即X1~f1, ..., Xm~fm
                for i = 1 : Problem.M
                    relevance = Rele_var{i};
                    % Kriging model for fi
                    [mX, mY]   = dsmerge(A1Dec(:,relevance), A1Obj(:,i));
                    dmodel     = dacefit(mX,mY,'regpoly1','corrgauss',THETA(i,relevance),...
                        1e-5.*ones(1,length(relevance)),100.*ones(1,length(relevance)));
                    Model{i}   = dmodel;
                    THETA(i,relevance) = dmodel.theta;
                end
                
                % 共轭模型，即[f2...fm(X1-joint)~f1,...,[f1...fm-1(Xm-joint)]~fm               
                for i = 1 : M
                    relevance = Rele_var{i};
                    diff_X = setdiff(relevance,joint);
                    THETA_id = D+1:D+M-1;
                    THETA_id = [diff_X,THETA_id];
                    conj_Y = [1:i-1,i+1:M];
                    X_train = [A1Dec(:,diff_X),A1Obj(:,conj_Y)];
                    [mX, mY]   = dsmerge(X_train, A1Obj(:,i));
                    dmodel     = dacefit(mX,mY,'regpoly1','corrgauss',THETA_conj(i,THETA_id),1e-5.*ones(1,length(THETA_id)),100.*ones(1,length(THETA_id)));
                    Model_conj{i} = dmodel;
                    THETA_conj(i,THETA_id) = dmodel.theta;
                end
                
                %% optimization
                PopDec = A1Dec;
                w      = 1;
                while w <= wmax
                    drawnow('limitrate');
                    OffDec = OperatorGA(Problem,PopDec);
                    PopDec = [PopDec;OffDec];
                    [N,~]  = size(PopDec);
                    
                    %% prediction
                    % 降维后的模型预测
                    PopObj = zeros(N,Problem.M);
                    MSE    = zeros(N,Problem.M);
                    for i = 1: M
                        relevance = Rele_var{i};
                        for j = 1 : N
                            [PopObj(j,i),~,MSE(j,i)] = predictor(PopDec(j,relevance),Model{i});
                        end
                    end
                    % 共轭模型预测，基于上一步模型预测的结果
                    for i = 1 : M
                        relevance = Rele_var{i};
                        diff_X = setdiff(relevance,joint);
                        conj_Y = [1:i-1,i+1:M];
                        for j = 1 : N
                            X_pred = [PopDec(j,diff_X),PopObj(j,conj_Y)];
                            [popobj,~,mse] = predictor(X_pred,Model_conj{i});
                            if mse < MSE(j,i)
                                PopObj(j,i) = popobj;
                                MSE(j,i) = mse;
                            end
                        end
                    end
                    
                    index  = KEnvironmentalSelection(PopObj,V,(w/wmax)^alpha);
                    PopDec = PopDec(index,:);
                    PopObj = PopObj(index,:);
                    % Adapt referece vectors
                    if ~mod(w,ceil(wmax*0.1))
                        V(1:Problem.N,:) = V0.*repmat(max(PopObj,[],1)-min(PopObj,[],1),size(V0,1),1);
                    end
                    w = w + 1; 
                end

                % Select mu solutions for re-evaluation
                [NumVf,~] = NoActive(A1Obj,V0);
                PopNew    = KrigingSelect(PopDec,PopObj,MSE(index,:),V,V0,NumVf,0.05*Problem.N,mu,(w/wmax)^alpha); 
                New       = Problem.Evaluation(PopNew);
                A2        = [A2,New];
                A1        = UpdataArchive(A1,New,V,mu,NI); 
            end
        end
    end
end